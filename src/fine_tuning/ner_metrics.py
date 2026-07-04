"""
ner_metrics.py — Entity-level precision/recall/F1 for NER models.

We roll our own instead of using `seqeval` because seqeval 1.2.2 has
a broken pyproject.toml that fails to install on Python 3.12 (the
library hasn't been actively maintained since 2020).

The semantics here match seqeval's default "strict" evaluation mode:
an entity is a maximal span starting with B-X and continuing with
I-X of the same type. A prediction is correct only if its type and
its span (start, end) exactly match a gold entity.

Public functions:
    extract_entities(labels)  -> list[(type, start, end)]
    entity_scores(true, pred) -> dict of {precision, recall, f1, ...}
"""

from collections import defaultdict


def extract_entities(labels: list[str]) -> list[tuple[str, int, int]]:
    """
    Convert a BIO-tagged label sequence into entity spans.

    Args:
        labels: List of strings like ["O", "B-PER", "I-PER", "O", "B-ORG"].

    Returns:
        List of (entity_type, start_idx, end_idx) tuples, inclusive on both ends.
        Malformed sequences (e.g. I-X without a preceding B-X) are handled
        leniently: an I-X that opens an entity is treated as B-X.
    """
    entities: list[tuple[str, int, int]] = []
    current_type: str | None = None
    current_start: int | None = None

    def close_current(end_idx: int) -> None:
        nonlocal current_type, current_start
        if current_type is not None:
            entities.append((current_type, current_start, end_idx))
            current_type = None
            current_start = None

    for i, label in enumerate(labels):
        if label == "O" or label == "":
            close_current(i - 1)
            continue

        # BIO labels look like "B-PER" or "I-PER"
        prefix, _, entity_type = label.partition("-")

        if prefix == "B":
            close_current(i - 1)
            current_type = entity_type
            current_start = i
        elif prefix == "I":
            if current_type == entity_type:
                # continuation of the current entity — do nothing
                pass
            else:
                # I-X either starts a new entity (malformed) or switches type
                close_current(i - 1)
                current_type = entity_type
                current_start = i
        else:
            # Unknown prefix — treat as O
            close_current(i - 1)

    # Close any entity still open at the end of the sequence
    close_current(len(labels) - 1)
    return entities


def entity_scores(
    true_labels: list[list[str]],
    pred_labels: list[list[str]],
) -> dict:
    """
    Compute entity-level precision, recall, F1 (micro-averaged across all sentences)
    plus a per-entity-type breakdown.

    Args:
        true_labels: List of label sequences (one per sentence), ground truth.
        pred_labels: Same shape as true_labels, model predictions.

    Returns:
        {
            "precision": micro-avg precision,
            "recall":    micro-avg recall,
            "f1":        micro-avg F1,
            "per_type":  { entity_type: {precision, recall, f1, support} },
        }
    """
    assert len(true_labels) == len(pred_labels), "Mismatched sentence count"

    # Aggregate TP / FP / FN globally AND per entity type
    tp_total, fp_total, fn_total = 0, 0, 0
    tp_by_type: dict[str, int] = defaultdict(int)
    fp_by_type: dict[str, int] = defaultdict(int)
    fn_by_type: dict[str, int] = defaultdict(int)

    for true_seq, pred_seq in zip(true_labels, pred_labels):
        true_entities = set(extract_entities(true_seq))
        pred_entities = set(extract_entities(pred_seq))

        true_positives = true_entities & pred_entities
        false_positives = pred_entities - true_entities
        false_negatives = true_entities - pred_entities

        tp_total += len(true_positives)
        fp_total += len(false_positives)
        fn_total += len(false_negatives)

        for etype, *_ in true_positives:
            tp_by_type[etype] += 1
        for etype, *_ in false_positives:
            fp_by_type[etype] += 1
        for etype, *_ in false_negatives:
            fn_by_type[etype] += 1

    def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

    p, r, f1 = prf(tp_total, fp_total, fn_total)

    all_types = set(tp_by_type) | set(fp_by_type) | set(fn_by_type)
    per_type = {}
    for etype in sorted(all_types):
        tp_i, fp_i, fn_i = tp_by_type[etype], fp_by_type[etype], fn_by_type[etype]
        p_i, r_i, f1_i = prf(tp_i, fp_i, fn_i)
        per_type[etype] = {
            "precision": p_i,
            "recall": r_i,
            "f1": f1_i,
            "support": tp_i + fn_i,          # # gold entities of this type
        }

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "per_type": per_type,
    }


def format_report(scores: dict) -> str:
    """Nice-looking per-class table, seqeval-style."""
    lines = []
    lines.append(f"{'type':>10s}  {'precision':>10s}  {'recall':>7s}  {'f1':>7s}  {'support':>8s}")
    lines.append("-" * 52)
    for etype, m in scores["per_type"].items():
        lines.append(
            f"{etype:>10s}  {m['precision']:>10.4f}  {m['recall']:>7.4f}  "
            f"{m['f1']:>7.4f}  {m['support']:>8d}"
        )
    lines.append("-" * 52)
    lines.append(
        f"{'micro':>10s}  {scores['precision']:>10.4f}  {scores['recall']:>7.4f}  {scores['f1']:>7.4f}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test on a small handcrafted example
    true = [
        ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"],
        ["O", "B-ORG", "I-ORG", "O"],
    ]
    pred = [
        ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"],   # perfect
        ["O", "B-ORG", "O", "O"],                          # missed the I-ORG continuation
    ]
    scores = entity_scores(true, pred)
    print("Self-test:")
    print(format_report(scores))
    # Expected:
    #   Sentence 1: 3 correct entities (PER, ORG, LOC)
    #   Sentence 2: predicted ORG as single-token instead of 2-token → wrong span
    #                so 0 TP, 1 FP, 1 FN
    #   Global: TP=3, FP=1, FN=1
    #   P = 3/4 = 0.75, R = 3/4 = 0.75, F1 = 0.75