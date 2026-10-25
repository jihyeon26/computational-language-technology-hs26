"""Read the Stage 1 inputs of the knowledge graph revision: human labels and evidence sentences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRIPLE = ["subject_id", "predicate", "object_id"]


def triple_key(subject_id: str, predicate: str, object_id: str) -> str:
    """One key per relation; `competes with` is symmetric, so both orders share a key."""
    if predicate == "competes with":
        subject_id, object_id = sorted([subject_id, object_id])
    return f"{subject_id}|{predicate}|{object_id}"


def with_key(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(key=[triple_key(*t) for t in frame[TRIPLE].itertuples(index=False, name=None)])


def load_review_labels(review_dir: Path | str) -> pd.DataFrame:
    """Labels of the Stage 1 relation review, one row per labelled relation.

    `random sample`: the 90 Tier B relations drawn at random, plus the supplementary relations of the graph
    (role-based typing), all labelled by people. `findings check`: the relations behind the Stage 1 findings, where
    FALSE labels were confirmed by a person and TRUE labels mostly come from an AI first pass (`label_source`).
    """
    review_dir = Path(review_dir)
    columns = {"head_id": "subject_id", "tail_id": "object_id", "relation": "predicate", "review_error_type": "error_type"}
    random_sample = pd.read_excel(review_dir / "relation_review_corrected_seed2026.xlsx", sheet_name="Review")
    supplement = pd.read_excel(review_dir / "relation_review_supplement_seed2026.xlsx", sheet_name="Review")
    supplement = supplement[supplement["added_by"].eq("role_typed_tier_B")]
    findings = pd.read_csv(review_dir / "insight_relation_check.csv")
    labels = pd.concat([
        random_sample.rename(columns=columns).assign(label=random_sample["review_correct"], label_set="random sample",
                                                     label_source="human"),
        supplement.rename(columns=columns).assign(label=supplement["review_correct"], label_set="random sample",
                                                  label_source="human"),
        findings.rename(columns=columns).assign(label=findings["verdict"], label_set="findings check",
                                                label_source=np.where(findings["human_checked"], "human", "AI first pass")),
    ], ignore_index=True)
    labels["label"] = labels["label"].astype(str).str.strip().str.upper()
    return with_key(labels)[["key", *TRIPLE, "label", "label_set", "label_source", "error_type"]]


def article_evidence(evidence_status: pd.DataFrame) -> pd.DataFrame:
    """GLiREL evidence sentences of Stage 1 with the corrected node IDs, one row per sentence and relation."""
    evidence = (evidence_status[evidence_status["evidence_status"].eq("ok")]
                .rename(columns={"head_id": "subject_id", "relation": "predicate", "tail_id": "object_id",
                                 "head": "subject", "tail": "object", "score": "glirel_score"}))
    evidence = with_key(evidence).assign(text_unit_id=lambda d: d["doc_id"] + ":" + d["sentence_id"].astype(str))
    return evidence[["key", *TRIPLE, "subject", "object", "doc_id", "text_unit_id", "sentence_text", "glirel_score"]]
