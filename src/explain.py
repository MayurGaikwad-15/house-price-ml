"""
explain.py — SHAP explainability for the best house-price model.

Produces:
  reports/figures/shap_summary.png   — bar chart of mean |SHAP| per feature
  reports/figures/shap_beeswarm.png  — beeswarm dot plot

Usage:
  python -m src.explain
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── colour palette matching EstateIQ dark theme ──────────────────────────────
ACCENT_1  = "#6366f1"   # indigo
ACCENT_2  = "#06b6d4"   # cyan
BG        = "#0a0d14"
SURFACE   = "#111827"
TEXT_PRI  = "#f1f5f9"
TEXT_MUT  = "#94a3b8"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    SURFACE,
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   TEXT_MUT,
    "xtick.color":       TEXT_MUT,
    "ytick.color":       TEXT_PRI,
    "text.color":        TEXT_PRI,
    "grid.color":        "#1e293b",
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.size":         11,
})

OUT_DIR = "reports/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load artefact ─────────────────────────────────────────────────────────────
print("[*] Loading model artefact...")
artifact      = joblib.load("models/model.pkl")
pipeline      = artifact["pipeline"]
X_train_trans = artifact["X_train_trans"]   # numpy array, already preprocessed
feature_names = artifact["feature_names"]
model_name    = artifact.get("best_model_name", "Unknown")

print(f"[*] Best model : {model_name}")
print(f"[*] Features   : {len(feature_names)}")
print(f"[*] Train rows : {X_train_trans.shape[0]}")

# ── Convert sparse to dense if needed ────────────────────────────────────────
if hasattr(X_train_trans, "toarray"):
    X_dense = X_train_trans.toarray()
else:
    X_dense = np.array(X_train_trans)

# ── Pick explainer ────────────────────────────────────────────────────────────
raw_model = pipeline.named_steps["model"]

if model_name in ("RandomForest", "XGBoost"):
    print("[*] Using shap.TreeExplainer...")
    explainer  = shap.TreeExplainer(raw_model)
    shap_vals  = explainer.shap_values(X_dense)
    # RandomForest returns a list for multi-output; take index 0 if needed
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
else:
    print("[*] Using shap.LinearExplainer...")
    explainer  = shap.LinearExplainer(raw_model, X_dense)
    shap_vals  = explainer.shap_values(X_dense)

shap_expl = shap.Explanation(
    values          = shap_vals,
    base_values     = explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray)
                      else explainer.expected_value[0],
    data            = X_dense,
    feature_names   = feature_names,
)

# ─────────────────────────────────────────────────────────────────────────────
#  TOP-N for readability
# ─────────────────────────────────────────────────────────────────────────────
TOP_N = 20
mean_abs = np.abs(shap_vals).mean(axis=0)
top_idx  = np.argsort(mean_abs)[-TOP_N:][::-1]

top_shap_vals     = shap_vals[:, top_idx]
top_X_dense       = X_dense[:, top_idx]
top_feature_names = [feature_names[i] for i in top_idx]

top_expl = shap.Explanation(
    values        = top_shap_vals,
    base_values   = shap_expl.base_values,
    data          = top_X_dense,
    feature_names = top_feature_names,
)

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 1 — Bar summary  (mean |SHAP| per feature)
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Generating SHAP bar summary plot...")

fig1, ax1 = plt.subplots(figsize=(10, 7))
fig1.patch.set_facecolor(BG)
ax1.set_facecolor(SURFACE)

vals_sorted   = mean_abs[top_idx]
feats_sorted  = top_feature_names

# Gradient fill: most important = indigo, least = cyan
cmap_colors = [mcolors.to_rgba(ACCENT_2, 0.85),
               mcolors.to_rgba(ACCENT_1, 0.95)]
n = len(vals_sorted)
bar_colors = [
    mcolors.to_rgba(
        mcolors.to_hex(
            [a + (b - a) * (i / max(n - 1, 1))
             for a, b in zip(mcolors.to_rgb(ACCENT_2), mcolors.to_rgb(ACCENT_1))]
        ), 0.9
    )
    for i in range(n)
]

bars = ax1.barh(
    range(n), vals_sorted[::-1],
    color=bar_colors, height=0.68,
    edgecolor="none"
)
ax1.set_yticks(range(n))
ax1.set_yticklabels(feats_sorted[::-1], fontsize=10, color=TEXT_PRI)
ax1.set_xlabel("Mean |SHAP value|  (impact on log-price)", color=TEXT_MUT, fontsize=11)
ax1.set_title(
    f"Feature Importance — {model_name}\n(SHAP Mean Absolute Impact, Top {TOP_N} Features)",
    color=TEXT_PRI, fontsize=13, fontweight="bold", pad=14
)
ax1.axvline(0, color="#1e293b", linewidth=1)
ax1.spines[:].set_visible(False)
ax1.grid(axis="x", alpha=0.3)

# Value labels on bars
for bar, val in zip(bars, vals_sorted[::-1]):
    ax1.text(
        val + max(vals_sorted) * 0.01, bar.get_y() + bar.get_height() / 2,
        f"{val:.4f}", va="center", ha="left", fontsize=8.5, color=TEXT_MUT
    )

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "shap_summary.png")
fig1.savefig(out1, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print(f"   Saved -> {out1}")

# ─────────────────────────────────────────────────────────────────────────────
#  PLOT 2 — Beeswarm
# ─────────────────────────────────────────────────────────────────────────────
print("[*] Generating SHAP beeswarm plot...")

# shap.plots.beeswarm writes to current figure; we control the figure wrapper
fig2 = plt.figure(figsize=(11, 8))
fig2.patch.set_facecolor(BG)

shap.plots.beeswarm(
    top_expl,
    max_display=TOP_N,
    show=False,
    color=plt.get_cmap("cool"),   # indigo-cyan gradient for feature values
)

# Override shap's default white background
ax2 = plt.gca()
ax2.set_facecolor(SURFACE)
ax2.spines[:].set_color("#1e293b")
ax2.tick_params(colors=TEXT_PRI)
ax2.xaxis.label.set_color(TEXT_MUT)

plt.title(
    f"SHAP Beeswarm — {model_name}  (Top {TOP_N} Features)\n"
    "Colour = feature value  ·  x-axis = SHAP impact on log-price",
    color=TEXT_PRI, fontsize=12, fontweight="bold", pad=12
)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "shap_beeswarm.png")
fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"   Saved -> {out2}")

print("\n[OK] Explainability artefacts ready in reports/figures/")
