"""
agent.py — Custom ReAct-style agent loop for FinSight.

Why custom instead of LangChain's create_react_agent:
    Mistral 7B Instruct v0.3 doesn't have native tool-calling training,
    so LangChain's create_react_agent (which expects a tool-calling
    chat model) can't reliably force it to emit structured tool calls.
    Instead, we prompt Mistral to output ACTION lines in a simple format
    and parse them ourselves.

The tools themselves are still LangChain BaseTool objects (from @tool).
The agent just uses .name, .description, .args, and .invoke() from them.

Public API:
    agent = FinSightAgent(tools, llm)
    trace = agent.run("What is Apple's revenue growth?")
    print(trace.final_answer)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are FinSight, a financial research agent that answers questions about US public company SEC filings.

You have access to the following tools:

{tool_descriptions}

To answer a question, you MUST use these tools to look up information from SEC filings. Do NOT rely on your general knowledge — always cite what you retrieve.

You respond in a strict format. On each turn output exactly ONE of these:

Format for calling a tool:
THOUGHT: <one sentence explaining your plan>
ACTION: <tool_name>
ARGS: <JSON object with the tool arguments>

Format for giving the final answer:
THOUGHT: <one sentence about what you've learned>
FINAL: <your complete answer to the user's question, citing any [Source N] references from the tool outputs>

Rules:
- ARGS must be valid JSON on a single line.
- Do not output anything after ARGS or after FINAL.
- If a tool result is enough to answer the question, produce FINAL immediately.
- If you need more information, call another tool.
- After 5 tool calls, you must produce FINAL even if incomplete.
"""

USER_TURN_TEMPLATE = "USER QUESTION: {question}"

OBSERVATION_TEMPLATE = "OBSERVATION (result of {tool_name}):\n{result}"


# ─────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────

@dataclass
class AgentStep:
    """One turn of the loop."""
    step_index: int
    thought: str = ""
    action: str = ""             # tool name (empty if final)
    args: dict = field(default_factory=dict)
    observation: str = ""
    is_final: bool = False
    raw_output: str = ""


@dataclass
class AgentTrace:
    """The full run: all steps, plus the final answer."""
    question: str
    steps: list[AgentStep]
    final_answer: str


# ─────────────────────────────────────────────────────────────────
# Parsing the LLM output
# ─────────────────────────────────────────────────────────────────

_THOUGHT_RE = re.compile(r"THOUGHT:\s*(.+?)(?=\n(?:ACTION|FINAL|ARGS|$))", re.S)
_ACTION_RE = re.compile(r"ACTION:\s*(\w+)", re.S)
_ARGS_RE = re.compile(r"ARGS:\s*(\{.*?\})\s*(?:\n|$)", re.S)
_FINAL_RE = re.compile(r"FINAL:\s*(.+)$", re.S)


def _parse_step(text: str) -> tuple[str, str | None, dict, str | None]:
    """
    Parse one LLM output into (thought, action, args, final_answer).

    Returns a tuple where either `action` (with `args`) or `final_answer`
    is populated — but not both. If the LLM's output can't be parsed at all,
    we return the raw text as `final_answer` (graceful degradation).
    """
    thought_m = _THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip() if thought_m else ""

    final_m = _FINAL_RE.search(text)
    if final_m:
        return thought, None, {}, final_m.group(1).strip()

    action_m = _ACTION_RE.search(text)
    args_m = _ARGS_RE.search(text)

    if action_m:
        action = action_m.group(1).strip()
        args_str = args_m.group(1).strip() if args_m else "{}"
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            # Malformed JSON — fall back to empty args
            args = {}
        return thought, action, args, None

    # Nothing parseable — treat the whole output as a final answer
    return thought, None, {}, text.strip()


# ─────────────────────────────────────────────────────────────────
# The agent
# ─────────────────────────────────────────────────────────────────

class FinSightAgent:
    """
    ReAct-style agent that uses LangChain tools + a custom loop.

    Usage:
        agent = FinSightAgent(tools=[...], llm=chat_model)
        trace = agent.run("Compare Apple and Microsoft on cybersecurity.")
        print(trace.final_answer)
    """

    def __init__(
        self,
        tools: list,
        llm,
        max_iterations: int = 5,
        verbose: bool = True,
    ):
        """
        Args:
            tools: List of LangChain BaseTool objects (from @tool).
            llm: A callable that takes a prompt string and returns a string.
                 Use a LangChain chat model wrapped so its invoke() returns str,
                 or pass any callable(str) -> str.
            max_iterations: Cap on tool calls per question.
            verbose: Print each step's output for debugging.
        """
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _describe_tools(self) -> str:
        """Format each tool's name, description, and args for the system prompt."""
        lines = []
        for name, tool in self.tools.items():
            # tool.args gives us the parameter schema
            args_hint = ", ".join(tool.args.keys())
            lines.append(f"- {name}({args_hint}): {tool.description.strip()}")
        return "\n".join(lines)

    def _call_llm(self, prompt: str) -> str:
        """
        Call the underlying LLM. We accept several forms of `llm`:
        - a LangChain chat model (has .invoke() returning an AIMessage)
        - a plain callable that takes a str, returns a str
        """
        if hasattr(self.llm, "invoke"):
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content if hasattr(response, "content") else str(response)
        return self.llm(prompt)

    def run(self, question: str) -> AgentTrace:
        """
        Run the ReAct loop on a question.

        Returns an AgentTrace with every intermediate step and the final answer.
        """
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            tool_descriptions=self._describe_tools()
        )
        user_turn = USER_TURN_TEMPLATE.format(question=question)

        # We maintain a running conversation as a single string; simpler than
        # message-list plumbing and works well for a single-turn agent.
        conversation = f"{system_prompt}\n\n{user_turn}\n"
        steps: list[AgentStep] = []

        for i in range(self.max_iterations + 1):
            # Prompt the LLM to take the next step
            raw_output = self._call_llm(conversation)

            if self.verbose:
                print(f"\n─── Step {i} — LLM output ───")
                print(raw_output)

            thought, action, args, final = _parse_step(raw_output)

            step = AgentStep(
                step_index=i,
                thought=thought,
                action=action or "",
                args=args,
                is_final=(final is not None),
                raw_output=raw_output,
            )

            if final is not None:
                # Done — the model produced a final answer
                steps.append(step)
                return AgentTrace(question=question, steps=steps, final_answer=final)

            if action not in self.tools:
                # Model tried to call something that doesn't exist
                observation = f"ERROR: unknown tool '{action}'. Available tools: {list(self.tools)}"
            else:
                tool = self.tools[action]
                try:
                    observation = tool.invoke(args)
                except Exception as e:
                    observation = f"ERROR calling {action}: {type(e).__name__}: {e}"

            step.observation = observation
            steps.append(step)

            if self.verbose:
                print(f"\n─── Step {i} — {action}({args}) ───")
                print(observation[:500] + ("..." if len(observation) > 500 else ""))

            # Append everything to the conversation and loop
            conversation += (
                f"\n{raw_output}\n\n"
                f"{OBSERVATION_TEMPLATE.format(tool_name=action, result=observation)}\n"
            )

        # Hit iteration cap without FINAL — return whatever the last output was
        return AgentTrace(
            question=question,
            steps=steps,
            final_answer="[Agent hit max_iterations without producing FINAL. See trace for partial results.]",
        )