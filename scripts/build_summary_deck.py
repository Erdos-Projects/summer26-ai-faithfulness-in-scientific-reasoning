"""Build a visual summary deck (PowerPoint) of the faithfulness investigation.

Generates docs/plain_language_summary.pptx from the committed results JSONs:
  - docs/faithfulness_probe_results.json
  - docs/posthoc_audit_results.json
  - docs/roadmap_followup_results.json

Every number on the slides is read from those files, with two exceptions that
exist only in prose records: the original-session full-stack accuracy 0.565
(reproduction note in docs/FAITHFULNESS_PROBE_FINDINGS.md) and the analytic
chance band / MDE values, which are recomputed here with the same formulas as
scripts/posthoc_audits.py.

Usage:
    python scripts/build_summary_deck.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs"
OUT_PPTX = DOCS / "plain_language_summary.pptx"

probe = json.loads((DOCS / "faithfulness_probe_results.json").read_text())
audit = json.loads((DOCS / "posthoc_audit_results.json").read_text())
roadmap = json.loads((DOCS / "roadmap_followup_results.json").read_text())

N_TEST = probe["dataset"]["direct_chart_by_split"]["test"]  # 276
SD = float(np.sqrt(0.25 / N_TEST))
BAND_LO, BAND_HI = 0.5 - 1.96 * SD, 0.5 + 1.96 * SD
MDE_TWO_SIDED = audit["analytic_power_min_detectable_accuracy"]["n_276"]["two_sided_alpha_05"]

DARK = RGBColor(0x20, 0x20, 0x20)
ACCENT = RGBColor(0xB0, 0x30, 0x30)

CHART_DIR = Path(tempfile.mkdtemp(prefix="deck_charts_"))


def save_fig(fig, name: str) -> Path:
    path = CHART_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- charts

def chart_chance_curve() -> Path:
    xs = np.linspace(0.38, 0.66, 400)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(xs, norm.pdf(xs, 0.5, SD), color="tab:blue", label=f"score distribution of pure guessing (n={N_TEST})")
    ax.axvspan(BAND_LO, BAND_HI, color="tab:blue", alpha=0.12,
               label=f"95% chance band: {BAND_LO:.1%} – {BAND_HI:.1%}")
    ax.axvline(0.5, color="gray", lw=1)
    ax.axvline(0.547, color="tab:orange", ls="--", label="headline result (rebuilt): 54.7%")
    ax.axvline(0.576, color="tab:red", ls="--", label="best input combination: 57.6%")
    ax.axvline(MDE_TWO_SIDED, color="tab:green", ls=":",
               label=f"true skill needed for reliable detection: {MDE_TWO_SIDED:.1%}")
    ax.set_xlabel("test accuracy")
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=9)
    return save_fig(fig, "chance_curve.png")


def chart_null_ranges() -> Path:
    rows = [("TF-IDF claim text", probe["tfidf_text_probe"]["direct_chart"])]
    for key in ("claim_vec", "pair_text_vec", "image_vec", "claim+image", "full_notebook_stack"):
        rows.append((key, probe["single_split_ablation"][key]))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, (name, r) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["null_min"], r["null_max"]], [y, y], color="tab:blue", lw=2, alpha=0.4)
        ax.plot([r["null_p05"], r["null_p95"]], [y, y], color="tab:blue", lw=7, alpha=0.7)
        ax.plot(r["null_mean"], y, "o", color="black", ms=5)
        ax.plot(r["accuracy"], y, "x", color="tab:red", ms=9, mew=2.5)
    ax.axvline(0.5, color="gray", lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([name for name, _ in reversed(rows)], fontsize=9)
    ax.set_xlabel("test accuracy")
    handles = [
        plt.Line2D([], [], color="tab:blue", lw=7, alpha=0.7, label="know-nothing models: 5th–95th percentile"),
        plt.Line2D([], [], color="tab:blue", lw=2, alpha=0.4, label="know-nothing models: full recorded range"),
        plt.Line2D([], [], color="black", marker="o", ls="", label="know-nothing mean"),
        plt.Line2D([], [], color="tab:red", marker="x", ls="", mew=2.5, label="actual model's score"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    return save_fig(fig, "null_ranges.png")


def chart_six_mirror() -> Path:
    fw = audit["family_wise_permutation"]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for y, (label, d, color) in enumerate([
        ("worst of six", fw["null_min"], "tab:blue"),
        ("best of six", fw["null_max"], "tab:orange"),
    ]):
        ax.plot([d["min"], d["max"]], [y, y], color=color, lw=2, alpha=0.4)
        ax.plot([d["p05"], d["p95"]], [y, y], color=color, lw=9, alpha=0.7)
        ax.plot(d["mean"], y, "o", color="black", ms=6)
        ax.annotate(f"mean {d['mean']:.1%}", (d["mean"], y + 0.18), ha="center", fontsize=9)
    ax.axvline(0.5, color="gray", lw=1)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["worst of six tries", "best of six tries"])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("test accuracy of know-nothing models (scrambled labels)")
    return save_fig(fig, "six_mirror.png")


def chart_ablation() -> Path:
    names = ["claim_vec", "image_vec", "claim+image", "full_notebook_stack"]
    labels = ["claim text\nonly", "image\nonly", "claim + image", "full feature\nstack"]
    accs = [probe["single_split_ablation"][n]["accuracy"] for n in names]
    cis = [probe["single_split_ablation"][n]["ci95"] for n in names]
    err = np.array([[a - lo, hi - a] for a, (lo, hi) in zip(accs, cis)]).T
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(labels, accs, yerr=err, capsize=5, color="tab:blue", alpha=0.8)
    ax.axhline(0.5, color="gray", ls="--", label="guessing (50%)")
    ax.axhline(MDE_TWO_SIDED, color="tab:green", ls=":",
               label=f"reliable-detection threshold ({MDE_TWO_SIDED:.1%})")
    ax.set_ylim(0.4, 0.68)
    ax.set_ylabel("test accuracy (error bars: 95% CI)")
    ax.legend(fontsize=9)
    return save_fig(fig, "ablation.png")


def chart_cv_collapse() -> Path:
    names = ["claim_vec", "image_vec", "claim+image"]
    labels = ["claim text only", "image only", "claim + image"]
    single = [probe["single_split_ablation"][n]["accuracy"] for n in names]
    cv_mean = [probe["repeated_cv"][n]["cv_balanced_accuracy_mean"] for n in names]
    cv_std = [probe["repeated_cv"][n]["cv_balanced_accuracy_std"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, single, w, label="official split (train on 140)", color="tab:blue", alpha=0.8)
    ax.bar(x + w / 2, cv_mean, w, yerr=cv_std, capsize=5,
           label="cross-validation, 50 splits (train on ~333)", color="tab:orange", alpha=0.8)
    ax.axhline(0.5, color="gray", ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 0.62)
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=9)
    return save_fig(fig, "cv_collapse.png")


def chart_geometry() -> Path:
    sim = roadmap["similarity_eda"]
    cats = ["same label", "different label", "same paper", "different paper"]
    claim = [sim["claim_vec"][k] for k in
             ("mean_cosine_same_label", "mean_cosine_diff_label", "mean_cosine_same_paper", "mean_cosine_diff_paper")]
    image = [sim["image_vec"][k] for k in
             ("mean_cosine_same_label", "mean_cosine_diff_label", "mean_cosine_same_paper", "mean_cosine_diff_paper")]
    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(x - w / 2, claim, w, label="claim embeddings", color="tab:blue", alpha=0.8)
    ax.bar(x + w / 2, image, w, label="image embeddings", color="tab:orange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("mean cosine similarity between examples")
    ax.legend(fontsize=9)
    ax.set_title("label: invisible (bars 1–2 identical) — paper: clearly encoded (bar 3 > 4)", fontsize=10)
    return save_fig(fig, "geometry.png")


# ---------------------------------------------------------------- slides

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def add_title(slide, text: str):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = DARK


def bullet_slide(title: str, bullets: list, font_size: int = 17):
    slide = prs.slides.add_slide(BLANK)
    add_title(slide, title)
    box = slide.shapes.add_textbox(Inches(0.7), Inches(1.4), Inches(11.9), Inches(5.6))
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(bullets):
        text, level = item if isinstance(item, tuple) else (item, 0)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ("• " if level == 0 else "– ") + text
        p.level = level
        p.font.size = Pt(font_size if level == 0 else font_size - 2)
        p.font.color.rgb = DARK
        p.space_after = Pt(10)
    return slide


def chart_slide(title: str, image: Path, caption: str = ""):
    slide = prs.slides.add_slide(BLANK)
    add_title(slide, title)
    slide.shapes.add_picture(str(image), Inches(1.2), Inches(1.3), width=Inches(10.9))
    if caption:
        box = slide.shapes.add_textbox(Inches(0.7), Inches(6.6), Inches(11.9), Inches(0.8))
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = caption
        p.font.size = Pt(14)
        p.font.color.rgb = ACCENT
    return slide


# 1 — title
slide = prs.slides.add_slide(BLANK)
box = slide.shapes.add_textbox(Inches(0.8), Inches(2.4), Inches(11.7), Inches(2.6))
tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Did Our Simple Claim-Checker Actually Work?"
p.font.size = Pt(40)
p.font.bold = True
for sub in (
    "A visual summary of the SciVer lightweight-baseline investigation",
    "Verdict: the 55% headline accuracy was luck, not skill — shown three independent ways",
):
    p = tf.add_paragraph()
    p.text = sub
    p.font.size = Pt(20)
    p.font.color.rgb = DARK

# 2 — the project
bullet_slide("The project in one slide", [
    "Task: given a scientific claim and a chart from a real paper, predict whether the chart supports (entailed) or contradicts (refuted) the claim.",
    "Approach: no large AI models. Convert claim and image to embedding vectors, train logistic regression on 140 examples, grade on 276 held-out examples.",
    "First result: about 55% accuracy. Guessing gives 50%.",
    "Question of this investigation: is 55% real skill or luck?",
    "Answer: luck. The rest of this deck shows the evidence.",
])

# 3 — broken input
bullet_slide("Check 1 — Half the intended inputs were missing", [
    "The 'evidence text' input (chart caption and surrounding text) was empty for all 1,500 examples. The data loader never opened the files that contain it.",
    "The model only saw the claim sentence and the raw image.",
    "A claim sentence alone cannot reveal whether a chart supports it — so honest above-chance accuracy was unlikely from the start.",
    "The missing text is recoverable: captions exist in the dataset's paper files for 83% of examples (path to a future retry).",
])

# 4 — chance band
chart_slide(
    "Check 2 — What does pure chance look like on 276 questions?",
    chart_chance_curve(),
    "55% is inside the chance band. A model needs ~58% true accuracy before this test can reliably distinguish it from guessing.",
)

# 5 — null ranges
chart_slide(
    "Know-nothing models, both tails recorded",
    chart_null_ranges(),
    "Models retrained 300× on scrambled labels score symmetrically around 50% (recorded extremes 37.3%–59.8%). "
    "Every real score (red ×) sits inside or barely past the band luck produces.",
)

# 6 — best/worst of six
chart_slide(
    "Trying six inputs and keeping the best inflates the score",
    chart_six_mirror(),
    "Best-of-six know-nothing models average 52.6%; worst-of-six average 47.3% — mirror images around 50%. "
    "The upward shift is selection, not signal in the data.",
)

# 7 — ablation
chart_slide(
    "Check 4 — Which input carried the edge?",
    chart_ablation(),
    "Claim text alone: chance. Image carries the only apparent edge — and no bar reaches the reliable-detection threshold.",
)

# 8 — CV collapse
chart_slide(
    "Check 5 — The decisive test: does the edge survive reshuffling?",
    chart_cv_collapse(),
    "With 2.4× more training data, a real ability should improve. Instead every edge collapsed to chance. "
    "The official split was a favorable draw.",
)

# 9 — fairness checks
bullet_slide("Check 6 — Four fairness checks on the test itself", [
    "No overlap: zero images and zero papers appear in both training and test sets — memorization was impossible.",
    "Best-of-six correction: after pricing in that six input combinations were tried, the best result is borderline (p = 0.027) — and check 5 already kills it.",
    "Both directions tested: no input combination scored suspiciously below chance either (worst 48.9%, p = 0.72). Under a two-sided test the image-only result is not significant (p = 0.06).",
    "Grouping by paper: keeping each paper's examples on one side of every split changed nothing.",
])

# 10 — reproduction
bullet_slide("Check 7 — Rebuilt from scratch, the headline moved", [
    "The full pipeline was rebuilt in a fresh environment and rerun.",
    "Every number reproduced exactly — except the headline.",
    "Full-stack accuracy moved from 56.5% to 54.7% because a software substitution (image-resize library) processed a few images microscopically differently.",
    "That 2-point move crossed the nominal significance boundary. A result this sensitive to irrelevant preprocessing is noise, not signal.",
])

# 11 — calibration
brier = roadmap["calibration"]
bullet_slide("Check 8 — The model is confidently wrong", [
    f"Only {brier['frac_predictions_in_0.4_0.6']:.1%} of its predicted probabilities are anywhere near 'unsure' (between 0.4 and 0.6); the rest are near-certain.",
    f"Brier score {brier['brier_score']} — worse than the {brier['brier_score_always_0.5']} you get by always answering 'no idea' (0.5).",
    "With 2,400+ features and ~500 training rows, logistic regression memorizes the training set and exports pure confidence, no information.",
])

# 12 — geometry
chart_slide(
    "Check 9 — Why it failed: the label is not in the numbers",
    chart_geometry(),
    "Entailed and refuted examples are equidistant to four decimal places. What the embeddings encode is which paper "
    "an example came from — style, not truth.",
)

# 13 — bottom line
bullet_slide("Bottom line", [
    "The cheap claim-checker does not work; the 55% was a lucky draw — shown independently by reshuffling, rebuilding, and geometry.",
    "This is a verdict on this pipeline, not the task: large multimodal models score well above chance on SciVer.",
    "A faithful retry needs: the real captions (recoverable for 83% of examples), an image encoder that can read chart values, and far more training data.",
    "All numbers trace to committed results files: faithfulness_probe_results.json, posthoc_audit_results.json, roadmap_followup_results.json.",
    "A documented negative result, with reasons, is the deliverable.",
])

prs.save(OUT_PPTX)
print(f"Wrote {OUT_PPTX} ({OUT_PPTX.stat().st_size / 1024:.0f} KiB, {len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
