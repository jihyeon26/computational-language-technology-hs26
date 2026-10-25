"""Check whether an evidence sentence states a graph relation, with a natural language inference (NLI) model.

Each (sentence, relation) pair becomes premise = sentence and hypothesis = the relation written as a sentence
("Qwen was developed by Alibaba."). The entailment probability is the support score of that sentence for that
relation. The hypothesis uses the node names of the graph, not the words of the sentence, so a sentence about
"Llama Nemotron" does not support a relation of the node "Llama".
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd

HYPOTHESES = {
    "developed by": "{h} was developed by {t}.",
    "released by": "{h} was released by {t}.",
    "announced by": "{h} was announced by {t}.",
    "member of family": "{h} is a model of the {t} family.",
    "successor of": "{h} is the successor of {t}.",
    "based on": "{h} is based on {t}.",
    "fine-tuned from": "{h} was fine-tuned from {t}.",
    "distilled from": "{h} was distilled from {t}.",
    "quantized version of": "{h} is a quantized version of {t}.",
    "has parameter count": "{h} has {t}.",
    "has context window": "{h} has a context window of {t}.",
    "licensed under": "{h} is licensed under {t}.",
    "has openness type": "{h} is {t}.",
    "uses architecture": "{h} uses {t}.",
    "uses technique": "{h} uses {t}.",
    "has capability": "{h} has {t} capabilities.",
    "evaluated on": "{h} was evaluated on {t}.",
    "outperforms": "{h} performs better than {t}.",
    "competes with": "{h} competes with {t}.",
    "trained on data": "{h} was trained on {t}.",
    "trained using hardware": "{h} was trained using {t}.",
    "runs on": "{h} runs on {t}.",
    "manufactured by": "{h} is made by {t}.",
    "available on": "{h} is available on {t}.",
    "integrated into": "{h} is integrated into {t}.",
    "used by": "{h} is used by {t}.",
    "supported by": "{h} is supported by {t}.",
    "subsidiary of": "{h} is part of {t}.",
    "acquired by": "{h} was acquired by {t}.",
    "invested in": "{h} invested in {t}.",
    "partner of": "{h} is a partner of {t}.",
    "founded by": "{h} was founded by {t}.",
    "works for": "{h} works for {t}.",
    "headquartered in": "{h} is based in {t}.",
    "complies with": "{h} complies with {t}.",
    "regulated by": "{h} is regulated by {t}.",
}
SYMMETRIC = {"competes with", "partner of"}


def hypothesis_name(canonical_name: str, sentence: str) -> str:
    """Node name used in the hypothesis.

    Lexicon families can have composite names ("Command R / Aya"); the part that occurs in the sentence is used,
    otherwise the first part. Underscores of normalised names (united_states) become spaces.
    """
    parts = [p.strip() for p in str(canonical_name).replace("_", " ").split(" / ") if p.strip()]
    lowered = str(sentence).lower()
    return next((p for p in parts if p.lower() in lowered), parts[0] if parts else str(canonical_name))


def build_pairs(evidence: pd.DataFrame, head_name: str = "head_name", tail_name: str = "tail_name",
                premise: str = "sentence_text") -> pd.DataFrame:
    """One premise/hypothesis pair per evidence row; symmetric relations get both orders."""
    unknown = sorted(set(evidence["predicate"]) - set(HYPOTHESES))
    if unknown:
        raise ValueError(f"no hypothesis template for: {unknown}")
    rows = []
    for row in evidence.itertuples():
        text = getattr(row, premise)
        head = hypothesis_name(getattr(row, head_name), text)
        tail = hypothesis_name(getattr(row, tail_name), text)
        template = HYPOTHESES[row.predicate]
        orders = [("forward", head, tail)] + ([("reverse", tail, head)] if row.predicate in SYMMETRIC else [])
        for direction, h, t in orders:
            rows.append({"row_index": row.Index, "direction": direction, "premise": str(text),
                         "hypothesis": template.format(h=h, t=t)})
    return pd.DataFrame(rows)


class NLIVerifier:
    """Entailment probabilities of a sequence-classification NLI model (premise, hypothesis).

    The model is loaded on the first prediction, so a run served entirely from the cache needs no model.
    """

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 16, max_length: int = 512):
        self.model_name, self.device, self.batch_size, self.max_length = model_name, device, batch_size, max_length
        self.model = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device).eval()
        labels = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entailment, self.contradiction = labels["entailment"], labels["contradiction"]

    def predict(self, premises: list[str], hypotheses: list[str], progress: bool = True) -> np.ndarray:
        """Array (n, 2): entailment and contradiction probability."""
        if self.model is None:
            self._load()
        order = np.argsort([len(p) for p in premises])  # similar lengths in a batch -> less padding
        result = np.zeros((len(premises), 2), dtype=np.float32)
        batches = range(0, len(order), self.batch_size)
        if progress:
            from tqdm.auto import tqdm
            batches = tqdm(batches, desc="NLI", total=math.ceil(len(order) / self.batch_size))
        with self.torch.inference_mode():
            for start in batches:
                idx = order[start:start + self.batch_size]
                encoded = self.tokenizer([premises[i] for i in idx], [hypotheses[i] for i in idx], truncation="only_first",
                                         max_length=self.max_length, padding=True, return_tensors="pt").to(self.device)
                probs = self.torch.softmax(self.model(**encoded).logits.float(), dim=-1).cpu().numpy()
                result[idx, 0], result[idx, 1] = probs[:, self.entailment], probs[:, self.contradiction]
        return result


def _pair_key(model_name: str, premise: str, hypothesis: str) -> str:
    return hashlib.sha1(f"{model_name}\x1f{premise}\x1f{hypothesis}".encode()).hexdigest()


def score_pairs(pairs: pd.DataFrame, verifier: NLIVerifier | None, cache_path: Path | str, model_name: str) -> pd.DataFrame:
    """Add entailment / contradiction probabilities; pairs already in the cache file (.csv or .parquet) are not
    scored again."""
    cache_path = Path(cache_path)
    read, write = ((pd.read_csv, pd.DataFrame.to_csv) if cache_path.suffix == ".csv"
                   else (pd.read_parquet, pd.DataFrame.to_parquet))
    scored = pairs.copy()
    scored["pair_key"] = [_pair_key(model_name, p, h) for p, h in zip(scored["premise"], scored["hypothesis"])]
    cache = (read(cache_path) if cache_path.exists()
             else pd.DataFrame(columns=["pair_key", "entailment", "contradiction"]))
    missing = scored.loc[~scored["pair_key"].isin(cache["pair_key"]), ["pair_key", "premise", "hypothesis"]]
    missing = missing.drop_duplicates("pair_key")
    if len(missing):
        if verifier is None:
            raise ValueError(f"{len(missing)} pairs are not in the cache and no verifier was given")
        probs = verifier.predict(missing["premise"].tolist(), missing["hypothesis"].tolist())
        new = pd.DataFrame({"pair_key": missing["pair_key"].values, "entailment": probs[:, 0], "contradiction": probs[:, 1]})
        cache = pd.concat([cache, new], ignore_index=True)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write(cache, cache_path, index=False)
    return scored.merge(cache, on="pair_key", how="left", validate="many_to_one")


def aggregate_support(evidence: pd.DataFrame, scored_pairs: pd.DataFrame, keys: list[str],
                      threshold: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sentence-level support (best order of a symmetric relation) and edge-level support per `keys`.

    A sentence supports its relation when the entailment probability reaches `threshold`; an edge is supported when
    at least one of its sentences is.
    """
    best = (scored_pairs.sort_values("entailment", ascending=False).drop_duplicates("row_index")
            .set_index("row_index")[["entailment", "contradiction", "hypothesis"]])
    sentences = evidence.join(best.rename(columns={"entailment": "support_score", "contradiction": "contradiction_score"}))
    sentences["supported"] = sentences["support_score"] >= threshold
    edges = (sentences.groupby(keys)
             .agg(support_max=("support_score", "max"), support_mean=("support_score", "mean"),
                  scored_sentences=("support_score", "size"), supported_sentences=("supported", "sum"))
             .reset_index())
    edges["supported_sentences"] = edges["supported_sentences"].astype(int)
    edges["support_status"] = np.where(edges["supported_sentences"] > 0, "supported", "not_supported")
    return sentences, edges


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - margin, centre + margin)
