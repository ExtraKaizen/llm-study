"""
AMO-Bench Experiment Utilities
──────────────────────────────
Answer extraction, verification (latex2sympy + SymPy + LLM judge),
checkpointing, API calls with retry, and run-ID file naming.
"""

import copy
import json
import os
import time

from latex2sympy2_extended import latex2sympy
from sympy import simplify, Abs, N, solve, oo, zoo, nan, Symbol
from timeout_function_decorator import timeout
from openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════
# Answer Extraction
# ═══════════════════════════════════════════════════════════════════════

ANSWER_PREFIX_LIST = [
    "### the final answer is:", "### the final answer:",
    "### final answer is:", "### final answer:",
    "### the final answer is", "### the final answer",
    "### final answer is", "### final answer",
]
ANSWER_PREFIX_LIST_WO_HASH = [p[4:] for p in ANSWER_PREFIX_LIST]

THINK_POSTFIX_LIST = ["</think>", "</longcat_think>"]

CUT_LIST = ["\\medskip", "\n---", "\n##", "\n\n", "\n**", "\nExplanation", "\nNote:"]

REMOVE_LIST = [
    "\\bigl", "\\bigr", "\\Bigl", "\\Bigr",
    "\\biggl", "\\biggr", "\\Biggl", "\\Biggr",
    "\\bigg", "\\Bigg", "\\big", "\\Big",
    "\\left", "\\right",
]

REPLACE_LIST = [
    ["\u2018", "'"], ["\u2019", "'"],
    ["\u201c", '"'], ["\u201d", '"'],
    ["\uff08", "("], ["\uff09", ")"],
    ["\uff0c", ", "], ["\uff1a", ": "],
    ["\uff1b", "; "], ["\u3002", ". "],
    ["\uff01", "! "], ["\uff1f", "? "],
    ["\u2026", "..."], ["\u2013", "-"], ["\u2212", "-"],
]


def _extract_boxed(text):
    """Extract the last \\boxed{...} expression with proper nested-brace handling."""
    # Find all \boxed{ occurrences, take the last one
    pattern = "\\boxed{"
    idx = text.rfind(pattern)
    if idx == -1:
        return text

    start = idx + len(pattern)
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[idx:i + 1]  # Return full \boxed{...}
    # Unmatched brace — return from \boxed{ onward
    return text[idx:]


def pred_extractor(pred, answer_type):
    """Extract the predicted answer from a model response string."""
    pred_extract = pred.replace("\uff1a", ": ")

    for postfix in THINK_POSTFIX_LIST:
        pred_extract = pred_extract.split(postfix)[-1].strip()

    for prefix in ANSWER_PREFIX_LIST + ANSWER_PREFIX_LIST_WO_HASH:
        if prefix in pred_extract.lower():
            tail = pred_extract.lower().split(prefix)[-1]
            pred_extract = pred_extract[-len(tail):].strip()
            break

    if answer_type != "description":
        for pat in REMOVE_LIST:
            pred_extract = pred_extract.replace(pat, "")

    for pat, new in REPLACE_LIST:
        pred_extract = pred_extract.replace(pat, new)

    while " }" in pred_extract:
        pred_extract = pred_extract.replace(" }", "}")
    while ".}" in pred_extract:
        pred_extract = pred_extract.replace(".}", "}")

    if answer_type in ["number", "variable", "set"]:
        for s in ["\\,", "\\;", "\\,", "\\;"]:
            pred_extract = pred_extract.replace(s, "")
        pred_extract = pred_extract.replace("\n", " ")

    if answer_type in ["number", "variable"]:
        pred_extract = pred_extract.replace(",", "")
        pred_extract = (pred_extract
                        .replace("\\{", "(").replace("\\}", ")")
                        .replace("\\[", "(").replace("\\]", ")"))

    # Extract \boxed{...} — strips trailing junk that confuses latex parsers
    if answer_type in ["number", "variable", "set"] and "\\boxed{" in pred_extract:
        pred_extract = _extract_boxed(pred_extract)

    return pred_extract.strip()


def pred_cut(pred_extract):
    """Strip trailing separators for a second grading attempt."""
    for pat in CUT_LIST:
        pred_extract = pred_extract.split(pat)[0].strip()
    return pred_extract


# ═══════════════════════════════════════════════════════════════════════
# Verification Functions
# Uses latex2sympy directly — math_verify's multiprocessing breaks on Windows.
# ═══════════════════════════════════════════════════════════════════════

@timeout(30)
def _solve_with_timeout(exp):
    return solve(exp)


def _strip_boxed(text):
    """Remove \\boxed{} wrapper if present, returning inner content."""
    s = text.strip()
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[7:-1]
    return s


def _parse_latex(text):
    """Parse a LaTeX string into a SymPy expression. Returns None on failure."""
    s = _strip_boxed(text.strip())
    # Remove surrounding $ signs
    s = s.strip("$").strip()
    if not s:
        return None
    try:
        return latex2sympy(s)
    except Exception:
        return None


def _sym_equal(a, b, float_rounding=4):
    """Check if two SymPy expressions are mathematically equivalent."""
    if a is None or b is None:
        return False
    try:
        # Quick identity check
        if a == b:
            return True
        # Symbolic simplification
        diff = simplify(a - b)
        if diff == 0:
            return True
        # Numeric fallback
        diff_val = complex(N(diff))
        tol = 10 ** (-float_rounding)
        return abs(diff_val) < tol
    except Exception:
        # Last resort: string comparison of evaluated forms
        try:
            a_val = round(float(N(a)), float_rounding)
            b_val = round(float(N(b)), float_rounding)
            return a_val == b_val
        except Exception:
            return False


def verify_number_set_answer(pred_extract, info):
    """Verify a number or set answer using latex2sympy + SymPy."""
    pred_expr = _parse_latex(pred_extract)
    gold_expr = _parse_latex(info["answer"])

    result = _sym_equal(pred_expr, gold_expr)

    # If pred has "=" in it (e.g., "x = 42"), try just the RHS
    if not result and pred_extract and "=" in pred_extract:
        rhs = pred_extract.split("=")[-1].strip()
        rhs_expr = _parse_latex(rhs)
        result = _sym_equal(rhs_expr, gold_expr)

    return result


def verify_variable_answer(pred_extract, info):
    """Verify a variable expression by evaluating at multiple test points."""
    assert "try_list" in info

    pred_inner = _strip_boxed(pred_extract)
    gold_inner = _strip_boxed(info["answer"])

    # Handle \qquad / \quad separators
    if "\\qquad" in pred_inner:
        pred_inner = pred_inner.split("\\qquad")[-2].strip()
    if "\\quad" in pred_inner:
        pred_inner = pred_inner.split("\\quad")[-2].strip()

    pred_inner = pred_inner.split("=")[-1].strip()
    gold_inner = gold_inner.split("=")[-1].strip()

    pred_expr = _parse_latex(pred_inner)
    gold_expr = _parse_latex(gold_inner)

    if pred_expr is None or gold_expr is None:
        return False

    for try_str in info["try_list"]:
        # Build substitution dict from try_str like "n=5" or "a=2,b=3,c=4"
        subs = {}
        for part in try_str.split(","):
            var, val = part.split("=")
            subs[Symbol(var.strip())] = int(val.strip())

        try:
            pred_val = N(pred_expr.subs(subs))
            gold_val = N(gold_expr.subs(subs))
        except Exception:
            return False

        if not _sym_equal(pred_val, gold_val, float_rounding=8):
            return False

    return True


SCORE_PROMPT = (
    "For the following math problem, we have the reference answer and the student's answer.\n"
    "Determine whether the student's answer is equivalent to the reference answer.\n"
    'If equivalent, output "Correct".\n'
    'If not equivalent, output "Incorrect".\n\n'
    "### Problem\n{question}\n\n"
    "### Reference Answer\n{gold}\n\n"
    "### Student Answer\n{pred}\n\n"
    "Now, please provide your judgment.\n"
    "Please strictly follow the format below to summarize your conclusion at the end of your judgment:\n"
    "### Conclusion: Correct/Incorrect\n"
    "If the answer involves a decimal approximation, it must be accurate to at least four decimal places."
)


def verify_description_answer(pred_extract, info, api_key, base_url):
    """Verify a description answer using LLM-as-judge with 5-vote majority."""
    prompt = SCORE_PROMPT.format(
        question=info["prompt"], gold=info["answer"], pred=pred_extract
    )
    if len(prompt) >= 20000:
        prompt = prompt[:10000] + "\n\n...\n\n" + prompt[-10000:]

    client = OpenAI(api_key=api_key, base_url=base_url)
    votes = []
    for _ in range(5):
        try:
            resp = client.chat.completions.create(
                model="o4-mini",
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=8192,
                temperature=1.0,
            )
            conclusion = resp.choices[0].message.content.lower().split("conclusion:")[-1]
            vote = (
                "correct" in conclusion.split()
                and "not correct" not in conclusion
                and "n't correct" not in conclusion
            )
        except Exception:
            vote = False
        votes.append(vote)

    return votes.count(True) > votes.count(False)


# ═══════════════════════════════════════════════════════════════════════
# Unified Grading Pipeline
# ═══════════════════════════════════════════════════════════════════════

def _verify_result(pred_extract, info, api_key=None, base_url=None):
    """Route to the correct verifier based on answer_type."""
    if info["answer_type"] == "description":
        return verify_description_answer(pred_extract, info, api_key, base_url)
    elif info["answer_type"] in ["number", "set"]:
        return verify_number_set_answer(pred_extract, info)
    else:
        return verify_variable_answer(pred_extract, info)


def grade_response(pred, info, api_key=None, base_url=None):
    """
    Grade a single model response against the gold answer.

    Returns a dict with detailed grading info:
      - score: 1.0 or 0.0
      - gold_answer, answer_type
      - extracted_answer, extracted_answer_cut
      - verify_original, verify_cut
      - error (if any)
    """
    answer_type = info["answer_type"]
    pred_extract = pred_extractor(pred, answer_type)
    pred_extract_cut = pred_cut(pred_extract) if answer_type != "description" else None

    detail = {
        "gold_answer": info["answer"],
        "answer_type": answer_type,
        "extracted_answer": pred_extract,
        "extracted_answer_cut": pred_extract_cut,
        "verify_original": None,
        "verify_cut": None,
        "error": None,
    }

    try:
        result = _verify_result(pred_extract, info, api_key, base_url)
        detail["verify_original"] = bool(result)
    except Exception as e:
        detail["verify_original"] = False
        detail["error"] = f"original: {e}"
        result = False

    result_cut = False
    if answer_type != "description" and pred_extract_cut:
        try:
            result_cut = _verify_result(pred_extract_cut, info, api_key, base_url)
            detail["verify_cut"] = bool(result_cut)
        except Exception as e:
            detail["verify_cut"] = False
            detail["error"] = (detail["error"] or "") + f" | cut: {e}"

    detail["score"] = 1.0 if (result or result_cut) else 0.0
    return detail


# ═══════════════════════════════════════════════════════════════════════
# Dataset Helpers
# ═══════════════════════════════════════════════════════════════════════

def append_try_list(ori_info):
    """Attach test-value lists for variable-type problems (Q5, Q37)."""
    info = copy.deepcopy(ori_info)
    qid = info["question_id"]
    if qid == 5 and info["answer_type"] == "variable":
        info["try_list"] = [f"n={i}" for i in range(1, 21)]
    elif qid == 37 and info["answer_type"] == "variable":
        info["try_list"] = [f"a={i},b={i+1},c={i+2}" for i in range(2, 19)]
    return info


# ═══════════════════════════════════════════════════════════════════════
# Run-ID & File Paths
# ═══════════════════════════════════════════════════════════════════════

def make_run_id(model_name, prompt_name, temperature, reasoning_name):
    """Build a deterministic run identifier from experiment parameters."""
    temp_str = str(temperature).replace(".", "p")
    return f"{model_name}__{prompt_name}__t{temp_str}__{reasoning_name}"


def _model_from_run_id(run_id):
    """Extract the model name (first segment) from a run_id."""
    return run_id.split("__")[0]


def get_raw_path(run_id, base_dir="../results/raw_responses"):
    model = _model_from_run_id(run_id)
    return os.path.join(base_dir, model, f"{run_id}.jsonl")


def get_graded_path(run_id, base_dir="../results/graded_responses"):
    model = _model_from_run_id(run_id)
    return os.path.join(base_dir, model, f"{run_id}_graded.json")


# ═══════════════════════════════════════════════════════════════════════
# Checkpoint Helpers
# ═══════════════════════════════════════════════════════════════════════

def load_completed_ids(filepath):
    """Read a JSONL file and return the set of question_ids already saved."""
    done = set()
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def append_response(filepath, question_id, model_response,
                    reasoning_content=None, finish_reason=None, usage=None):
    """Append one response line to a JSONL file (creates dirs if needed).

    Stores model_response (the final answer text) plus optional metadata:
      - reasoning_content: the model's internal chain-of-thought (if reasoning enabled)
      - finish_reason: "stop" (complete) or "length" (truncated by max_tokens)
      - usage: token counts {prompt_tokens, completion_tokens, total_tokens}
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    record = {"question_id": question_id, "model_response": model_response}
    if reasoning_content is not None:
        record["reasoning_content"] = reasoning_content
    if finish_reason is not None:
        record["finish_reason"] = finish_reason
    if usage is not None:
        record["usage"] = usage
    with open(filepath, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


# ═══════════════════════════════════════════════════════════════════════
# API Call with Retry
# ═══════════════════════════════════════════════════════════════════════

def call_model(client, model_id, prompt, temperature, max_tokens,
               reasoning_enabled=None, max_retries=3):
    """Call an LLM via OpenAI-compatible API with exponential back-off.

    Returns a dict with:
      - content: the model's final answer text
      - reasoning_content: internal reasoning trace (None if reasoning disabled)
      - finish_reason: "stop" or "length" (truncated)
      - usage: {prompt_tokens, completion_tokens, total_tokens}
    """
    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if reasoning_enabled is not None:
                kwargs["extra_body"] = {"reasoning": {"enabled": reasoning_enabled}}

            response = client.chat.completions.create(**kwargs)
            message = response.choices[0].message
            choice = response.choices[0]

            # Extract reasoning trace (OpenRouter returns it differently per model)
            reasoning = None
            if hasattr(message, "reasoning_details") and message.reasoning_details:
                reasoning = message.reasoning_details
            elif hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning = message.reasoning_content
            # Some models nest it in a list of dicts
            if isinstance(reasoning, list):
                reasoning = "\n".join(
                    r.get("content", "") if isinstance(r, dict) else str(r)
                    for r in reasoning
                )

            # Token usage
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return {
                "content": message.content,
                "reasoning_content": reasoning,
                "finish_reason": choice.finish_reason,
                "usage": usage,
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise e
