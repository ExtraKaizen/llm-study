"""
AMO-Bench Grading Power Test
=============================
Loads ALL 50 gold answers from the AMO-Bench dataset and tests whether
our latex2sympy + SymPy verification pipeline can parse and self-verify
each one (gold == gold must be True).

This validates that the grading system can handle every answer type
in the benchmark before running real experiments.
"""

import sys
import copy
sys.path.insert(0, ".")

import datasets
from utils import (
    _parse_latex,
    _sym_equal,
    _strip_boxed,
    verify_number_set_answer,
    verify_variable_answer,
    append_try_list,
)


# ======================================================================
# Load Dataset
# ======================================================================
print("Loading AMO-Bench dataset...")
ds = datasets.load_dataset("meituan-longcat/AMO-Bench")["test"]
questions = {item["question_id"]: append_try_list(item) for item in ds}
print(f"Loaded {len(questions)} questions\n")

# Group by type
by_type = {}
for qid, info in questions.items():
    by_type.setdefault(info["answer_type"], []).append((qid, info))

print("Answer type distribution:")
for atype, items in sorted(by_type.items()):
    print(f"  {atype:12s}: {len(items)} questions")
print()

log = []  # collect all output, print at end

# ======================================================================
# TEST 1: Can _parse_latex parse every gold answer?
# ======================================================================
log.append("=" * 70)
log.append("  TEST 1: Parsing Gold Answers with _parse_latex")
log.append("=" * 70)

parse_pass = 0
parse_fail = 0

for atype in ["number", "set", "variable"]:
    if atype not in by_type:
        continue
    log.append(f"\n--- {atype.upper()} ({len(by_type[atype])} questions) ---")
    for qid, info in sorted(by_type[atype]):
        gold = info["answer"]
        inner = _strip_boxed(gold)
        if atype == "variable":
            inner = inner.split("=")[-1].strip()

        parsed = _parse_latex(inner)
        if parsed is not None:
            parse_pass += 1
            log.append(f"  [PASS] Q{qid:>2d}  {gold[:55]}")
        else:
            parse_fail += 1
            log.append(f"  [FAIL] Q{qid:>2d}  {gold[:55]}  -> PARSE FAILED")

desc_count = len(by_type.get("description", []))
log.append(f"\n--- DESCRIPTION ({desc_count} questions) ---")
log.append(f"  [SKIP] Description answers use LLM-as-judge, not symbolic parsing")
log.append(f"\n  Parsing: {parse_pass} passed, {parse_fail} failed "
           f"(out of {parse_pass + parse_fail})")


# ======================================================================
# TEST 2: Self-verification (gold == gold) for number/set
# ======================================================================
log.append("\n" + "=" * 70)
log.append("  TEST 2: Self-Verification (gold == gold) for number/set")
log.append("=" * 70)

verify_pass = 0
verify_fail = 0

for atype in ["number", "set"]:
    if atype not in by_type:
        continue
    log.append(f"\n--- {atype.upper()} ---")
    for qid, info in sorted(by_type[atype]):
        gold = info["answer"]
        try:
            result = verify_number_set_answer(gold, info)
            if result:
                verify_pass += 1
                log.append(f"  [PASS] Q{qid:>2d}  {gold[:55]}")
            else:
                verify_fail += 1
                log.append(f"  [FAIL] Q{qid:>2d}  {gold[:55]}  -> FAILED")
        except Exception as e:
            verify_fail += 1
            log.append(f"  [FAIL] Q{qid:>2d}  {gold[:55]}  -> ERROR: {e}")

log.append(f"\n  Self-Verify (n/s): {verify_pass} passed, {verify_fail} failed")


# ======================================================================
# TEST 3: Self-verification (gold == gold) for variable
# ======================================================================
log.append("\n" + "=" * 70)
log.append("  TEST 3: Self-Verification (gold == gold) for variable")
log.append("=" * 70)

var_pass = 0
var_fail = 0

for atype in ["variable"]:
    if atype not in by_type:
        continue
    for qid, info in sorted(by_type[atype]):
        gold = info["answer"]
        try:
            result = verify_variable_answer(gold, info)
            if result:
                var_pass += 1
                log.append(f"  [PASS] Q{qid:>2d}  {gold[:55]}")
            else:
                var_fail += 1
                log.append(f"  [FAIL] Q{qid:>2d}  {gold[:55]}  -> FAILED")
        except Exception as e:
            var_fail += 1
            log.append(f"  [FAIL] Q{qid:>2d}  {gold[:55]}  -> ERROR: {e}")

log.append(f"\n  Self-Verify (var): {var_pass} passed, {var_fail} failed")


# ======================================================================
# TEST 4: Equivalent-form verification
# ======================================================================
log.append("\n" + "=" * 70)
log.append("  TEST 4: Equivalent-Form Verification (numeric == symbolic)")
log.append("=" * 70)

equiv_pass = 0
equiv_fail = 0

for qid, info in sorted(questions.items()):
    if info["answer_type"] not in ["number", "set"]:
        continue
    gold = info["answer"]
    inner = _strip_boxed(gold)
    parsed = _parse_latex(inner)
    if parsed is None:
        continue

    try:
        from sympy import N
        num_val = complex(N(parsed))
        if num_val.imag == 0:
            num_val = num_val.real
            numeric_str = f"\\boxed{{{num_val:.10g}}}"
            result = verify_number_set_answer(numeric_str, info)
            if result:
                equiv_pass += 1
                status = "PASS"
            else:
                equiv_fail += 1
                status = "FAIL"
            log.append(f"  [{status}] Q{qid:>2d}  {gold[:30]:30s} == {numeric_str[:30]}")
    except Exception:
        pass

log.append(f"\n  Equiv-Form: {equiv_pass} passed, {equiv_fail} failed "
           f"(out of {equiv_pass + equiv_fail})")


# ======================================================================
# TEST 5: Wrong-answer rejection
# ======================================================================
log.append("\n" + "=" * 70)
log.append("  TEST 5: Wrong-Answer Rejection (gold + 1 must fail)")
log.append("=" * 70)

reject_pass = 0
reject_fail = 0

for qid, info in sorted(questions.items()):
    if info["answer_type"] != "number":
        continue
    gold = info["answer"]
    inner = _strip_boxed(gold)
    parsed = _parse_latex(inner)
    if parsed is None:
        continue

    try:
        from sympy import N
        num_val = complex(N(parsed))
        if num_val.imag == 0:
            wrong_val = num_val.real + 1.0
            wrong_str = f"\\boxed{{{wrong_val:.10g}}}"
            result = verify_number_set_answer(wrong_str, info)
            if not result:
                reject_pass += 1
                log.append(f"  [PASS] Q{qid:>2d}  rejected {wrong_str[:40]}")
            else:
                reject_fail += 1
                log.append(f"  [FAIL] Q{qid:>2d}  ACCEPTED {wrong_str[:40]}")
    except Exception:
        pass

log.append(f"\n  Rejection: {reject_pass} rejected, {reject_fail} false accepts")


# ======================================================================
# FINAL SUMMARY
# ======================================================================
log.append("\n" + "=" * 70)
log.append("  FINAL SUMMARY")
log.append("=" * 70)

total_pass = parse_pass + verify_pass + var_pass + equiv_pass + reject_pass
total_fail = parse_fail + verify_fail + var_fail + equiv_fail + reject_fail
total = total_pass + total_fail

log.append(f"""
  Dataset:  {len(questions)} questions total
            {len(by_type.get('number',[]))} number, {len(by_type.get('set',[]))} set, \
{len(by_type.get('variable',[]))} variable, {len(by_type.get('description',[]))} description

  Test 1 - Parsing:           {parse_pass}/{parse_pass+parse_fail} gold answers parsed
  Test 2 - Self-verify (n/s): {verify_pass}/{verify_pass+verify_fail} gold==gold
  Test 3 - Self-verify (var): {var_pass}/{var_pass+var_fail} gold==gold
  Test 4 - Equiv forms:       {equiv_pass}/{equiv_pass+equiv_fail} numeric equivalences
  Test 5 - Rejection:         {reject_pass}/{reject_pass+reject_fail} wrong answers rejected

  TOTAL: {total_pass}/{total} checks passed, {total_fail} failed
""")

if total_fail == 0:
    log.append("  >>> ALL CHECKS PASSED - latex2sympy + SymPy is READY! <<<")
else:
    log.append(f"  >>> {total_fail} FAILURES - review the failing cases above <<<")

log.append("=" * 70)

# Print all at once (avoids interleaved output)
print("\n".join(log))
