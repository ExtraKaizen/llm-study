# %% Cell 0: Install Dependencies
# !pip install matplotlib pandas numpy


# # AMO-Bench Experiment Visualizations
# Loads graded results and generates publication-quality plots.
# All figures saved to `../plots/`.

# %% Cell 1: Imports & Load Data
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GRADED_DIR = "../results/graded_responses"
PLOTS_DIR = "../plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

graded_files = sorted(glob.glob(os.path.join(GRADED_DIR, "**", "*_graded.json"), recursive=True))
records = []

for fp in graded_files:
    with open(fp, "r") as f:
        data = json.load(f)

    run_id = data["run_id"]
    parts = run_id.split("__")

    if len(parts) == 4:
        model, prompt, temp_str, reasoning = parts
        temp = float(temp_str.replace("t", "").replace("p", "."))
    else:
        model, prompt, temp, reasoning = run_id, "unknown", 0.0, "unknown"

    records.append({
        "run_id": run_id,
        "model": model,
        "prompt": prompt,
        "temperature": temp,
        "reasoning": reasoning,
        "accuracy": data["summary"]["accuracy"],
        "correct": data["summary"]["correct"],
        "total": data["summary"]["total"],
        "per_question": {  # Extract scores from detailed or flat format
            k: (v["score"] if isinstance(v, dict) else v)
            for k, v in (data.get("questions") or data.get("per_question") or {}).items()
        },
    })

df = pd.DataFrame(records)
print(f"Loaded {len(df)} graded runs\n")
print(df[["model", "prompt", "temperature", "reasoning", "accuracy"]].to_string(index=False))


# %% Cell 2: Accuracy by Model (Bar Chart)
fig, ax = plt.subplots(figsize=(10, 5))

model_acc = df.groupby("model")["accuracy"].mean().sort_values(ascending=False)
colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(model_acc)))

bars = ax.bar(model_acc.index, model_acc.values, color=colors,
              edgecolor="white", linewidth=1.2, width=0.6)

for bar, val in zip(bars, model_acc.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

ax.set_ylabel("Average Accuracy")
ax.set_title("Model Performance on AMO-Bench (Parser Subset)")
ax.set_ylim(0, min(1.0, model_acc.max() + 0.15))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_by_model.png"))
plt.show()


# %% Cell 3: Temperature Effect (Line Chart)
fig, ax = plt.subplots(figsize=(10, 5))

cmap = plt.cm.tab10
for i, model in enumerate(df["model"].unique()):
    subset = df[df["model"] == model].groupby("temperature")["accuracy"].mean()
    ax.plot(subset.index, subset.values, marker="o", linewidth=2.2,
            markersize=7, label=model, color=cmap(i))

ax.set_xlabel("Temperature")
ax.set_ylabel("Accuracy")
ax.set_title("Effect of Temperature on Accuracy")
ax.legend(frameon=True, fancybox=True, shadow=True, loc="best")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.grid(True, alpha=0.25, linestyle="--")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "temperature_effect.png"))
plt.show()


# %% Cell 4: Prompt Comparison (Grouped Bar)
fig, ax = plt.subplots(figsize=(9, 5))

pivot = df.groupby(["model", "prompt"])["accuracy"].mean().unstack("prompt")
pivot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=1.0, width=0.7)

ax.set_ylabel("Accuracy")
ax.set_title("Zero-Shot vs Chain-of-Thought by Model")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(title="Prompt", frameon=True, fancybox=True)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "prompt_comparison.png"))
plt.show()


# %% Cell 5: Reasoning Mode Comparison (Grouped Bar)
fig, ax = plt.subplots(figsize=(9, 5))

pivot = df.groupby(["model", "reasoning"])["accuracy"].mean().unstack("reasoning")
pivot.plot(kind="bar", ax=ax, edgecolor="white", linewidth=1.0, width=0.7)

ax.set_ylabel("Accuracy")
ax.set_title("Reasoning vs No-Reasoning by Model")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.legend(title="Mode", frameon=True, fancybox=True)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "reasoning_comparison.png"))
plt.show()


# %% Cell 6: Heatmap — Model × Temperature
fig, ax = plt.subplots(figsize=(10, max(3, len(df["model"].unique()) * 0.8 + 1)))

pivot = df.groupby(["model", "temperature"])["accuracy"].mean().unstack("temperature")
im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{t}" for t in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel("Temperature")
ax.set_title("Accuracy Heatmap: Model × Temperature")

for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        if not np.isnan(val):
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center",
                    color=color, fontsize=11, fontweight="bold")

fig.colorbar(im, ax=ax, label="Accuracy", shrink=0.8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "heatmap_model_temp.png"))
plt.show()


# %% Cell 7: Per-Problem Difficulty
all_per_q = {}
for rec in records:
    for qid_str, val in rec["per_question"].items():
        qid = int(qid_str)
        score = val["score"] if isinstance(val, dict) else val
        all_per_q.setdefault(qid, []).append(score)

problem_diff = sorted(
    [(qid, np.mean(accs)) for qid, accs in all_per_q.items()],
    key=lambda x: x[1],
)

fig, ax = plt.subplots(figsize=(14, 5))
qids = [f"Q{q}" for q, _ in problem_diff]
accs = [a for _, a in problem_diff]
colors = plt.cm.RdYlGn(accs)

ax.bar(qids, accs, color=colors, edgecolor="white", linewidth=0.5)
ax.set_ylabel("Solve Rate (across all runs)")
ax.set_title("Problem Difficulty — Sorted by Solve Rate")
ax.set_ylim(0, 1)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
ax.axhline(y=np.mean(accs), color="gray", linestyle="--", alpha=0.5, label=f"Mean: {np.mean(accs):.1%}")
ax.legend(frameon=True)
plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "problem_difficulty.png"))
plt.show()
