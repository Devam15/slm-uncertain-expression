"""
uncertain_expression_probe.py
===============================
Black-box confidence signal via PROMPT PERTURBATION, not hidden states.
Based on: Joo et al., "Black-Box Hallucination Detection via Consistency
Under the Uncertain Expression."

WHY THIS FILE EXISTS
---------------------
drift_probe.py needed white-box access (hidden states via transformers),
which your review flagged as a real problem: it can't work with closed
models, and exposing internals is extra attack surface on a deployed edge
device. This script needs NONE of that -- it only ever sends text prompts
and reads text answers, through the exact same Ollama setup router_demo.py
already uses. No transformers, no torch, no hidden layers.

THE IDEA, IN ONE LINE
-----------------------
self-consistency (router_demo.py) only varies RANDOMNESS (temperature) and
checks if the model agrees with itself. This method instead varies the
FRAMING of the question -- nudging the model to sound more certain or more
uncertain -- and checks if the model's actual answer changes. A model that
truly knows the fact should give the same answer regardless of framing. A
model that's guessing tends to wobble under a different framing, even when
it wouldn't wobble under plain resampling.

PIPELINE
--------
1. Ask the question normally -> r1 (the "reference" answer)
2. Ask again, but prefix the answer with an overconfident nudge
   ("It must be...") -> r2
3. Ask again, but prefix the answer with an underconfident nudge
   ("I am not sure but it could be...") -> r3
4. Score r1 against MedHallu's Ground Truth (same 0.55 threshold convention
   as router_demo.py / drift_probe.py) -> correct/incorrect label
5. Measure how similar r2 and r3 are to r1 (embedding similarity, same
   sentence-transformers model already used elsewhere in this project).
   Take the WORSE (minimum) of the two similarities as the consistency
   score -- this is the paper's "ensemble" variant, and was their
   best-performing config.
6. Low consistency = risky = escalate to the big model. High consistency =
   trust the small model.
7. Evaluate: AUROC (does the consistency score actually predict
   correctness?) and a rejection-accuracy sweep, same format as
   drift_probe.py so the two methods are directly comparable.

HONEST LIMITATION (carried over from the paper itself)
----------------------------------------------------------
The paper's own analysis found a "bedrock confident" subset -- answers that
stay wrong AND consistent even under this method -- where no signal they
tested (including token log-probabilities) reliably told confidently-wrong
apart from confidently-right. This method narrows self-consistency's blind
spot, it doesn't eliminate it. Say that honestly in your writeup.

USAGE
-----
Real run (needs `pip install ollama sentence-transformers pandas
--break-system-packages`, plus Ollama running with qwen2.5:1.5b pulled --
you already have all of this from router_demo.py):
    python3 uncertain_expression_probe.py            # 350 questions from data/

Dry run with synthetic data, no Ollama needed (verifies the scoring /
evaluation logic only):
    python3 uncertain_expression_probe.py --mock --n-rows 60
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:1.5b"
CORRECTNESS_THRESHOLD = 0.55   # matches router_demo.py / drift_probe.py
CONSISTENCY_THRESHOLD = 0.60   # below this similarity -> treated as "wobbled" -> escalate

CERTAIN_PREFIX = "The correct answer is definitely"
UNCERTAIN_PREFIX = "My uncertain best guess is"


# ---------------------------------------------------------------------------
# Step 1-3: build Qwen's chat format BY HAND, so we can inject a prefix
# straight into the assistant's turn -- instead of choosing between "chat
# quality" (let Ollama auto-template) and "literal continuation" (raw mode,
# no template), we get both by constructing the exact template Ollama would
# have built anyway, and inserting our prefix right where the model's own
# answer would normally start.
# ---------------------------------------------------------------------------
def build_chat_prompt(question: str, assistant_prefix: str = "") -> str:
    """Qwen2.5 uses the ChatML format:
        <|im_start|>system
        {system}<|im_end|>
        <|im_start|>user
        {question}<|im_end|>
        <|im_start|>assistant
        {assistant's answer starts here}

    Used with raw=True, so Ollama sends this exact string with no further
    reformatting -- the model just continues from wherever our text stops.
    Leaving assistant_prefix empty gives a normal, full-quality chat answer
    (used for the standard/reference answer). A non-empty prefix makes the
    model continue an answer that already starts with our chosen framing
    (used for the certain/uncertain variants) -- all THREE calls now go
    through the identical chat structure, so they're stylistically
    comparable, unlike before when the standard answer used one format and
    the framed answers used another.
    """
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{assistant_prefix}"
    )


def standard_prompt(question: str) -> str:
    return build_chat_prompt(question, "")


def certain_prompt(question: str) -> str:
    return build_chat_prompt(question, CERTAIN_PREFIX)


def uncertain_prompt(question: str) -> str:
    return build_chat_prompt(question, UNCERTAIN_PREFIX)


def generate_ollama(prompt: str, model: str = DEFAULT_MODEL, temperature: float = 0.0) -> str:
    """Single Ollama call. temperature=0.0 (greedy) since we're testing framing
    sensitivity here, not sampling randomness -- that's self-consistency's job,
    not this method's.

    raw=True is used for ALL THREE prompts now, since we build the full chat
    template ourselves in build_chat_prompt(). This gives every answer the
    same chat-trained quality (fixing the earlier style mismatch between the
    standard answer and the framed ones) while still letting us inject a
    prefix directly into the assistant's turn for the framing variants.

    "<|im_end|>" is added to the stop list because that's the model's own
    learned end-of-turn signal -- it should still emit it even without Ollama
    auto-applying the template, since that behavior is trained into the
    model's weights, not just added by the template wrapper.
    """
    import ollama

    response = ollama.generate(
        model=model,
        prompt=prompt,
        raw=True,
        options={
            "temperature": temperature,
            "num_predict": 150,
            "stop": ["<|im_end|>", "\nQuestion:", "\n\nQuestion:", "\nQ:"],
        },
    )
    return response["response"].strip()


# ---------------------------------------------------------------------------
# Step 4-5: scoring (correctness + consistency), reusing the project's
# existing embedding-similarity convention
# ---------------------------------------------------------------------------
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")  # same model used elsewhere in the project


def text_similarity(a: str, b: str, embedder) -> float:
    from sentence_transformers import util
    emb_a = embedder.encode(a, convert_to_tensor=True)
    emb_b = embedder.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()


def score_correctness(answer: str, ground_truth: str, embedder, threshold: float = CORRECTNESS_THRESHOLD):
    similarity = text_similarity(answer, ground_truth, embedder)
    return int(similarity >= threshold), similarity


def compute_consistency(r1: str, r2: str, r3: str, embedder):
    """Returns (sim_certain, sim_uncertain, ensemble_min).

    ensemble_min = the WORSE of the two comparisons -- if the answer wobbles
    under EITHER framing, that counts as inconsistent. Matches the paper's
    best-performing "ensemble" variant.
    """
    sim_certain = text_similarity(r1, r2, embedder)
    sim_uncertain = text_similarity(r1, r3, embedder)
    ensemble_min = min(sim_certain, sim_uncertain)
    return sim_certain, sim_uncertain, ensemble_min


# ---------------------------------------------------------------------------
# Step 6-7: build the dataset + escalation decision
# ---------------------------------------------------------------------------
def escalation_decision(consistency_score: float, threshold: float = CONSISTENCY_THRESHOLD) -> bool:
    """Return True (escalate to big model) if the answer wobbled too much
    under prompt perturbation."""
    return consistency_score < threshold


def build_dataset(pilot_df: pd.DataFrame, model: str, embedder, verbose: bool = True):
    records = []

    for i, row in pilot_df.iterrows():
        question = row["Question"]
        ground_truth = row["Ground Truth"]

        r1 = generate_ollama(standard_prompt(question), model)
        r2 = generate_ollama(certain_prompt(question), model)
        r3 = generate_ollama(uncertain_prompt(question), model)

        label, correctness_sim = score_correctness(r1, ground_truth, embedder)
        sim_certain, sim_uncertain, consistency = compute_consistency(r1, r2, r3, embedder)
        would_escalate = escalation_decision(consistency)

        records.append({
            "question": question,
            "r1_standard": r1,
            "r2_certain": r2,
            "r3_uncertain": r3,
            "label": label,                  # 1 = correct, 0 = incorrect
            "correctness_sim": correctness_sim,
            "sim_certain": sim_certain,
            "sim_uncertain": sim_uncertain,
            "consistency": consistency,       # higher = more stable under perturbation
            "would_escalate": would_escalate,
        })

        if verbose:
            tag = "CORRECT" if label else "WRONG"
            flag = "ESCALATE" if would_escalate else "keep"
            print(f"[{i+1}/{len(pilot_df)}] label={tag:7s} consistency={consistency:.3f} -> {flag}")

    return records


# ---------------------------------------------------------------------------
# Evaluation: AUROC + rejection-accuracy sweep (same format as drift_probe.py
# so the two methods can be compared side by side)
# ---------------------------------------------------------------------------
def evaluate(records):
    from sklearn.metrics import roc_auc_score

    labels = np.array([r["label"] for r in records])
    consistency_scores = np.array([r["consistency"] for r in records])

    metrics = {"n": len(records), "n_correct": int(labels.sum())}
    if len(np.unique(labels)) < 2:
        metrics["auroc"] = None
    else:
        # consistency should be HIGHER for correct answers, so it directly
        # works as a "predicts correct" score for roc_auc_score
        metrics["auroc"] = float(roc_auc_score(labels, consistency_scores))

    return metrics, labels, consistency_scores


def rejection_accuracy_curve(labels: np.ndarray, consistency_scores: np.ndarray,
                              coverage_levels=(1.0, 0.8, 0.6, 0.4, 0.2)):
    order = np.argsort(-consistency_scores)  # most consistent (most trusted) first
    n = len(labels)
    curve = []
    for coverage in coverage_levels:
        k = max(1, int(round(n * coverage)))
        kept_idx = order[:k]
        answer_correct_rate = float(np.mean(labels[kept_idx] == 1))
        curve.append({"coverage": coverage, "n_kept": k, "answer_correct_rate": answer_correct_rate})
    return curve


# ---------------------------------------------------------------------------
# Mock mode: synthetic data, no Ollama or sentence-transformers needed.
# Tests ONLY the pure-logic parts (evaluation, rejection sweep) -- same
# philosophy as drift_probe.py's --mock flag.
# ---------------------------------------------------------------------------
def make_mock_records(n_rows: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n_rows)
    # correct answers get higher consistency on average, incorrect ones lower
    # -- but with real overlap/noise, unlike a perfectly separable toy case
    base = np.where(labels == 1, 0.75, 0.55)
    consistency = np.clip(base + rng.normal(0, 0.15, size=n_rows), 0, 1)
    records = [
        {"label": int(labels[i]), "consistency": float(consistency[i])}
        for i in range(n_rows)
    ]
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Black-box uncertain-expression consistency probe")
    parser.add_argument("--pilot-csv", type=str, default="data/medhallu_pilot_subset_350.csv")
    parser.add_argument("--n-rows", type=int, default=350)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--consistency-threshold", type=float, default=CONSISTENCY_THRESHOLD)
    parser.add_argument("--output-json", type=str, default="results/uncertain_expression_results.json")
    parser.add_argument("--mock", action="store_true",
                         help="Skip Ollama/embedder calls; verify evaluation logic with synthetic data")
    args = parser.parse_args()
    Path("results").mkdir(exist_ok=True)

    if args.mock:
        print(f"[mock mode] generating {args.n_rows} synthetic records...")
        records = make_mock_records(args.n_rows)
    else:
        print(f"Loading pilot data from {args.pilot_csv} (first {args.n_rows} rows)...")
        pilot_df = pd.read_csv(args.pilot_csv).head(args.n_rows)

        print("Loading sentence-transformers embedder...")
        embedder = load_embedder()

        print(f"Querying {args.model} via Ollama: 3 calls per question "
              f"(standard, certain-framed, uncertain-framed)...")
        records = build_dataset(pilot_df, args.model, embedder)

        # Save full per-question records for later row-by-row comparison
        # against drift_probe_results.json / router_demo.py output
        Path("results/uncertain_expression_records.json").write_text(json.dumps(records, indent=2))

    print(f"\nDataset: {len(records)} examples")

    metrics, labels, consistency_scores = evaluate(records)
    print(f"AUROC: {metrics['auroc']}")

    print("\nRejection sweep -- escalate the LEAST consistent N%, check answer correctness on the rest:")
    curve = rejection_accuracy_curve(labels, consistency_scores)
    for point in curve:
        print(f"  coverage={point['coverage']*100:5.0f}%  n_kept={point['n_kept']:3d}  "
              f"answer_correct_rate={point['answer_correct_rate']*100:5.1f}%")

    n_escalate = int(np.sum(consistency_scores < args.consistency_threshold))
    print(f"\nWith consistency-threshold={args.consistency_threshold}: "
          f"{n_escalate}/{len(consistency_scores)} questions would escalate to the big model")

    results = {"config": vars(args), "metrics": metrics, "rejection_accuracy_curve": curve}
    Path(args.output_json).write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.output_json}")

    if metrics["n"] < 30:
        print("\n[!] WARNING: fewer than 30 examples. Treat this as a pipeline sanity check, "
              "not a real performance estimate.")


if __name__ == "__main__":
    sys.exit(main())
