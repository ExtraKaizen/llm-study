# %% Cell 0: Install Dependencies
# !pip install openai datasets sympy latex2sympy2_extended timeout-function-decorator tqdm


# # AMO-Bench Grading Pipeline
# Reads raw response JSONL files from `results/raw_responses/`,
# grades each using **latex2sympy + SymPy** (symbolic verification),
# and saves graded results to `results/graded_responses/`.

# %% Cell 1: Configuration & Imports
import sys
import json
import os
import glob
import copy
import datasets
from tqdm import tqdm

sys.path.insert(0, ".")
from utils import grade_response, append_try_list, get_graded_path

RAW_DIR = "../results/raw_responses"

# Only needed when grading description answers (all 50 problems).
# If only_parser was True during experiment, this is unused.
JUDGE_API_KEY = "YOUR_API_KEY_HERE"
JUDGE_BASE_URL = "https://api.openai.com/v1"


# %% Cell 2: Load Dataset
dataset = datasets.load_dataset("meituan-longcat/AMO-Bench")["test"]
question_info = {item["question_id"]: append_try_list(item) for item in dataset}
print(f"Loaded {len(question_info)} gold answers from AMO-Bench")


# %% Cell 3: Grade All Response Files
# Set to True to re-grade even if graded files already exist
FORCE_REGRADE = True

response_files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.jsonl"), recursive=True))
print(f"Found {len(response_files)} response files to grade\n")

all_summaries = []

for filepath in response_files:
    run_id = os.path.basename(filepath).replace(".jsonl", "")
    graded_path = get_graded_path(run_id)

    # Skip already-graded runs (unless forcing)
    if not FORCE_REGRADE and os.path.exists(graded_path):
        with open(graded_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        all_summaries.append({"run_id": run_id, **existing["summary"]})
        print(f"[SKIP] {run_id}  (already graded: {existing['summary']['accuracy']:.1%})")
        continue

    # Load raw responses
    with open(filepath, "r", encoding="utf-8") as f:
        responses = [json.loads(line) for line in f if line.strip()]

    print(f"[GRADING] {run_id}  ({len(responses)} responses)")

    per_question = {}
    for resp in tqdm(responses, desc=f"  {run_id}", leave=True):
        qid = resp["question_id"]
        info = copy.deepcopy(question_info[qid])

        if resp["model_response"].startswith("ERROR:"):
            per_question[str(qid)] = {
                "score": 0.0,
                "gold_answer": info["answer"],
                "answer_type": info["answer_type"],
                "extracted_answer": None,
                "extracted_answer_cut": None,
                "verify_original": None,
                "verify_cut": None,
                "error": "API error during generation",
            }
            continue

        detail = grade_response(
            resp["model_response"],
            info,
            api_key=JUDGE_API_KEY,
            base_url=JUDGE_BASE_URL,
        )
        per_question[str(qid)] = detail

    scores = [q["score"] for q in per_question.values()]
    total = len(scores)
    correct = int(sum(scores))
    accuracy = correct / total if total > 0 else 0.0

    graded_result = {
        "run_id": run_id,
        "summary": {
            "total": total,
            "correct": correct,
            "accuracy": round(accuracy, 4),
        },
        "questions": per_question,
    }

    os.makedirs(os.path.dirname(graded_path), exist_ok=True)
    with open(graded_path, "w", encoding="utf-8") as f:
        json.dump(graded_result, f, indent=2, ensure_ascii=False)

    all_summaries.append({"run_id": run_id, **graded_result["summary"]})
    print(f"  → {accuracy:.1%}  ({correct}/{total})\n")


# %% Cell 4: Summary Table
print(f"\n{'═' * 74}")
print(f"  {'GRADING SUMMARY':^70}")
print(f"{'═' * 74}")
print(f"  {'Run ID':<52} {'Accuracy':>8}  {'Score':>8}")
print(f"{'─' * 74}")

for s in sorted(all_summaries, key=lambda x: x["accuracy"], reverse=True):
    bar = "█" * int(s["accuracy"] * 20) + "░" * (20 - int(s["accuracy"] * 20))
    print(f"  {s['run_id']:<52} {s['accuracy']:>7.1%}  {s['correct']:>3}/{s['total']}  {bar}")

print(f"{'═' * 74}")

if all_summaries:
    best = max(all_summaries, key=lambda x: x["accuracy"])
    print(f"\n  Best: {best['run_id']}  →  {best['accuracy']:.1%}")
