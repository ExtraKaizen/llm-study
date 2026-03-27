# %% Cell 0: Install Dependencies
# !pip install openai datasets tqdm sympy latex2sympy2_extended timeout-function-decorator


# # AMO-Bench Experiment Runner
# Run multiple LLMs × prompts × temperatures × reasoning modes on AMO-Bench.

# Edit **Cell 1** to configure, then run cells sequentially.

# %% Cell 1: Configuration
# ═══════════════════════════════════════════════════════════════════════
# Edit this cell to configure your experiment.
# ═══════════════════════════════════════════════════════════════════════

CONFIG = {
    # ── API ───────────────────────────────────────────────────────────
    "api_key": "YOUR_API_KEY_HERE",
    "base_url": "https://openrouter.ai/api/v1",

    # ── Models ────────────────────────────────────────────────────────
    # Add/remove models here. Each entry needs an API `id` and a short `name`, found in OpenRouter.
    "models": [
        {"id": "x-ai/grok-4.1-fast", "name": "grok-4.1"},
        # {"id": "moonshotai/kimi-k2-thinking", "name": "Kimi-K2-Thinking"},
        # {"id": "google/gemini-3.1-flash-lite-preview", "name": "Gemini-3.1-Flash-Lite-Preview"},
        # {"id": "openai/gpt-oss-120b", "name": "gpt-oss-120b"},
        # {"id": "z-ai/glm-5", "name": "GLM-5"}
    ],

    # ── Prompts ───────────────────────────────────────────────────────
    # {problem} is replaced with the actual problem text at runtime.
    "prompts": {
        "zero_shot": (
            "{problem}\n\n"
            "Provide your final answer in the format:\n"
            "### The final answer is: $\\boxed{{answer}}$"
        ),
        "cot": (
            "Solve this problem step by step.\n\n"
            "{problem}\n\n"
            "Provide your final answer in the format:\n"
            "### The final answer is: $\\boxed{{answer}}$"
        ),
    },

    # ── Temperatures ──────────────────────────────────────────────────
    "temperatures": [0.0, 0.5, 0.7, 1.0],

    # ── Max Tokens ────────────────────────────────────────────────────
    "max_tokens": 32000,

    # ── Reasoning Modes ───────────────────────────────────────────────
    "reasoning_modes": {
        "no_reasoning": False,
        "with_reasoning": True,
    },

    # ── Problem Subset ────────────────────────────────────────────────
    # True  → 39 parser-based problems only (number, set, variable)
    # False → all 50 problems (requires LLM judge API for description answers)
    "only_parser": True,
    # "only_parser": False,
}


# %% Cell 2: Imports & Dataset Setup
import sys
import datasets
from openai import OpenAI

sys.path.insert(0, ".")
from utils import (
    make_run_id,
    get_raw_path,
    load_completed_ids,
    append_response,
    call_model,
    append_try_list,
)

dataset = datasets.load_dataset("meituan-longcat/AMO-Bench")["test"]
all_problems = list(dataset)

if CONFIG["only_parser"]:
    problems = [p for p in all_problems if p["answer_type"] in ["number", "set", "variable"]]
else:
    problems = all_problems

print(f"Loaded {len(problems)} / {len(all_problems)} problems")
print(f"  Types: { {t: sum(1 for p in problems if p['answer_type'] == t) for t in set(p['answer_type'] for p in problems)} }")

client = OpenAI(api_key=CONFIG["api_key"], base_url=CONFIG["base_url"])


# %% Cell 3: Build Run Matrix
runs = []
for model in CONFIG["models"]:
    for prompt_name, prompt_template in CONFIG["prompts"].items():
        for temp in CONFIG["temperatures"]:
            for reasoning_name, reasoning_flag in CONFIG["reasoning_modes"].items():
                run_id = make_run_id(model["name"], prompt_name, temp, reasoning_name)
                runs.append({
                    "run_id": run_id,
                    "model": model,
                    "prompt_name": prompt_name,
                    "prompt_template": prompt_template,
                    "temperature": temp,
                    "reasoning_name": reasoning_name,
                    "reasoning_flag": reasoning_flag,
                })

print(f"Run matrix: {len(CONFIG['models'])} models × {len(CONFIG['prompts'])} prompts "
      f"× {len(CONFIG['temperatures'])} temps × {len(CONFIG['reasoning_modes'])} reasoning "
      f"= {len(runs)} runs")
print(f"Total API calls: {len(runs)} × {len(problems)} = {len(runs) * len(problems)}\n")

for i, r in enumerate(runs, 1):
    filepath = get_raw_path(r["run_id"])
    done = len(load_completed_ids(filepath))
    status = f"({done}/{len(problems)})" if done > 0 else "(new)"
    print(f"  {i:>3}. {r['run_id']}  {status}")


# %% Cell 4: Run Experiments
for run_idx, run in enumerate(runs, 1):
    run_id = run["run_id"]
    filepath = get_raw_path(run_id)
    completed = load_completed_ids(filepath)
    remaining = [p for p in problems if p["question_id"] not in completed]

    print(f"\n{'═' * 70}")
    print(f"[Run {run_idx}/{len(runs)}] {run_id}")
    print(f"  Done: {len(completed)}/{len(problems)} | Remaining: {len(remaining)}")
    print(f"{'═' * 70}")

    if not remaining:
        print("  → Already complete, skipping.")
        continue

    for prob_idx, prob in enumerate(remaining, 1):
        prompt = run["prompt_template"].format(problem=prob["prompt"])

        try:
            result = call_model(
                client=client,
                model_id=run["model"]["id"],
                prompt=prompt,
                temperature=run["temperature"],
                max_tokens=CONFIG["max_tokens"],
                reasoning_enabled=run["reasoning_flag"],
            )
            append_response(
                filepath,
                prob["question_id"],
                model_response=result["content"],
                reasoning_content=result.get("reasoning_content"),
                finish_reason=result.get("finish_reason"),
                usage=result.get("usage"),
            )

            # Warn if response was truncated
            status = "✓"
            if result.get("finish_reason") == "length":
                status = "⚠ TRUNCATED"
            tokens = result.get("usage", {})
            tok_info = f"  ({tokens.get('completion_tokens', '?')} tok)" if tokens else ""
            print(f"  [{prob_idx}/{len(remaining)}] Q{prob['question_id']} {status}{tok_info}")

        except Exception as e:
            append_response(filepath, prob["question_id"], f"ERROR: {e}")
            print(f"  [{prob_idx}/{len(remaining)}] Q{prob['question_id']} ✗  {e}")

print(f"\n{'═' * 70}")
print("All runs complete.")
print(f"{'═' * 70}")
