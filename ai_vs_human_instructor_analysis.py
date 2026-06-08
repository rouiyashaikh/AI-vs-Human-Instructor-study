# -*- coding: utf-8 -*-
"""
================================================================================
AI vs Human Instructor: Comparative Study Analysis
================================================================================
MSc Human-Centred Artificial Intelligence — Bournemouth University (2023–2025)
Dissertation: "The Role of Artificial Intelligence in Enhancing Personalised
              Learning in Higher Education: A Comparative Study of AI and
              Human Instructors"
Author: Rouiya Mirza Shaikh
Supervisor: Dr. Mohammed Naiseh
Ethics ID: 57901 (Approved: 19/06/2024)
--------------------------------------------------------------------------------
This script reproduces the full statistical analysis pipeline from the
dissertation, including:
  - Data simulation based on published mean/SD values from the dissertation
  - Label encoding of categorical variables
  - Descriptive statistics (mean, standard deviation)
  - Shapiro-Wilk normality test
  - Levene's homogeneity of variance test
  - Independent samples t-test
  - Mann-Whitney U test
  - Visualisations for all 4 scenarios x 6 questions
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, ttest_ind
import warnings
warnings.filterwarnings('ignore')

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)

# ── Study Parameters ───────────────────────────────────────────────────────────
N_PARTICIPANTS = 15          # Number of participants (n=15, as per ethics form)
LIKERT_MIN     = 1           # Minimum Likert value
LIKERT_MAX     = 7           # Maximum Likert value (7-point scale used throughout)

# ── Colour Palette ─────────────────────────────────────────────────────────────
COLOR_AI    = "#5B9BD5"      # Blue — AI Instructor
COLOR_HUMAN = "#ED7D31"      # Orange — Human Instructor
COLOR_SIG   = "#FF4444"      # Red — statistically significant marker

# ==============================================================================
# SECTION 1: DATA SIMULATION
# All mean and SD values are taken directly from the dissertation results tables
# (Chapter 4). Simulated data is generated to reproduce equivalent statistical
# properties for code demonstration purposes.
# ==============================================================================

def simulate_likert(mean, sd, n=N_PARTICIPANTS, low=LIKERT_MIN, high=LIKERT_MAX):
    """
    Simulate Likert-scale responses matching a target mean and SD.
    Clipped to valid Likert range [1, 7] and rounded to nearest integer.
    """
    raw = np.random.normal(loc=mean, scale=sd, size=n)
    return np.clip(np.round(raw), low, high).astype(int)


# ------------------------------------------------------------------------------
# Scenario 1 — Lecture Delivery
# Columns: Question | AI Mean | AI SD | Human Mean | Human SD | t | p
# Source: Dissertation Table, Chapter 4, Scenario 1
# ------------------------------------------------------------------------------
scenario1_params = {
    "Q1_Initial_Thoughts":     {"ai": (5.46, 1.49), "human": (5.81, 1.10), "t": -1.40, "p": 0.165},
    "Q2_Effectiveness":        {"ai": (5.20, 1.23), "human": (5.61, 1.02), "t": -1.87, "p": 0.064},
    "Q3_Likelihood_Acceptance":{"ai": (5.04, 1.44), "human": (5.39, 1.11), "t": -1.42, "p": 0.157},
    "Q4_Engagement":           {"ai": (4.76, 1.47), "human": (5.33, 1.23), "t": -2.21, "p": 0.030},  # SIGNIFICANT
    "Q5_Competency":           {"ai": (4.69, 1.49), "human": (5.04, 1.54), "t": -1.21, "p": 0.230},
    "Q6_Accuracy":             {"ai": (4.91, 1.48), "human": (5.06, 1.47), "t": -0.52, "p": 0.603},
}

# Scenario 2 — Answering Questions
scenario2_params = {
    "Q1_Initial_Thoughts":     {"ai": (4.83, 1.53), "human": (5.44, 1.09), "t": -2.39, "p": 0.0185},  # SIGNIFICANT
    "Q2_Effectiveness":        {"ai": (5.44, 1.64), "human": (6.09, 1.10), "t": -2.40, "p": 0.0179},  # SIGNIFICANT
    "Q3_Likelihood_Acceptance":{"ai": (5.13, 1.53), "human": (5.59, 1.12), "t": -1.79, "p": 0.0761},
    "Q4_Engagement":           {"ai": (4.56, 2.19), "human": (5.48, 1.29), "t": -1.31, "p": 0.1917},
    "Q5_Competency":           {"ai": (4.91, 1.61), "human": (5.28, 1.31), "t": -1.31, "p": 0.1917},
    "Q6_Accuracy":             {"ai": (4.96, 1.58), "human": (5.20, 1.29), "t": -0.87, "p": 0.3879},
}

# Scenario 3 — Providing Feedback
scenario3_params = {
    "Q1_Initial_Thoughts":     {"ai": (4.87, 1.48), "human": (5.13, 1.57), "t": -0.88, "p": 0.379},
    "Q2_Effectiveness":        {"ai": (5.11, 1.56), "human": (5.30, 1.72), "t": -0.59, "p": 0.560},
    "Q3_Likelihood_Acceptance":{"ai": (4.98, 1.69), "human": (5.28, 1.72), "t": -0.90, "p": 0.368},
    "Q4_Engagement":           {"ai": (4.96, 1.55), "human": (5.30, 1.71), "t": -1.06, "p": 0.292},
    "Q5_Competency":           {"ai": (4.69, 1.78), "human": (4.87, 1.90), "t": -0.52, "p": 0.603},
    "Q6_Accuracy":             {"ai": (4.52, 2.07), "human": (4.89, 1.73), "t": -1.01, "p": 0.316},
}

# Scenario 4 — Assignment Marking
scenario4_params = {
    "Q1_Initial_Thoughts":     {"ai": (4.91, 1.71), "human": (5.07, 1.60), "t": -0.52, "p": 0.602},
    "Q2_Effectiveness":        {"ai": (5.09, 1.65), "human": (5.20, 1.66), "t": -0.35, "p": 0.728},
    "Q3_Likelihood_Acceptance":{"ai": (4.89, 1.69), "human": (5.26, 1.71), "t": -1.13, "p": 0.260},
    "Q4_Engagement":           {"ai": (4.94, 1.66), "human": (5.22, 1.68), "t": -0.86, "p": 0.390},
    "Q5_Competency":           {"ai": (4.74, 1.78), "human": (4.94, 1.76), "t": -0.60, "p": 0.552},
    "Q6_Accuracy":             {"ai": (4.74, 1.75), "human": (5.11, 1.63), "t": -1.14, "p": 0.258},
}

ALL_SCENARIOS = {
    "Scenario 1: Lecture Delivery":        scenario1_params,
    "Scenario 2: Answering Questions":     scenario2_params,
    "Scenario 3: Providing Feedback":      scenario3_params,
    "Scenario 4: Assignment Marking":      scenario4_params,
}

# Question labels for axis display
Q_LABELS = {
    "Q1_Initial_Thoughts":      "Initial\nThoughts",
    "Q2_Effectiveness":         "Effectiveness",
    "Q3_Likelihood_Acceptance": "Likelihood of\nAcceptance",
    "Q4_Engagement":            "Engagement",
    "Q5_Competency":            "Competency",
    "Q6_Accuracy":              "Accuracy",
}


# ==============================================================================
# SECTION 2: GENERATE SIMULATED DATASET
# ==============================================================================

print("=" * 70)
print("GENERATING SIMULATED DATASET")
print("=" * 70)

rows = []
for scenario_name, params in ALL_SCENARIOS.items():
    for q_key, vals in params.items():
        ai_scores    = simulate_likert(*vals["ai"])
        human_scores = simulate_likert(*vals["human"])
        for i in range(N_PARTICIPANTS):
            rows.append({
                "Participant_ID":  i + 1,
                "Scenario":        scenario_name,
                "Question":        q_key,
                "Instructor_Type": "AI",
                "Score":           ai_scores[i],
            })
            rows.append({
                "Participant_ID":  i + 1,
                "Scenario":        scenario_name,
                "Question":        q_key,
                "Instructor_Type": "Human",
                "Score":           human_scores[i],
            })

df = pd.DataFrame(rows)

# ── Label Encoding (as described in dissertation Section 3.3) ──────────────────
# "Label encoding was used to translate categorical variables"
df["Instructor_Type_Encoded"] = df["Instructor_Type"].map({"AI": 0, "Human": 1})
df["Gender_Encoded"]          = 0   # Placeholder: Female=0, Male=1 per dissertation
df["Scenario_Encoded"]        = df["Scenario"].factorize()[0]

print(f"\nDataset shape: {df.shape}")
print(f"Participants : {N_PARTICIPANTS}")
print(f"Scenarios   : {df['Scenario'].nunique()}")
print(f"Questions   : {df['Question'].nunique()}")
print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))


# ==============================================================================
# SECTION 3: DESCRIPTIVE STATISTICS
# "Mean and Standard Deviation were calculated for each categorical response"
# ==============================================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS — ALL SCENARIOS")
print("=" * 70)

desc_rows = []
for scenario_name, params in ALL_SCENARIOS.items():
    print(f"\n{scenario_name}")
    print("-" * 60)
    print(f"{'Question':<30} {'AI Mean':>8} {'AI SD':>7} {'H Mean':>8} {'H SD':>7}")
    print("-" * 60)
    for q_key, vals in params.items():
        ai_m, ai_sd   = vals["ai"]
        h_m,  h_sd    = vals["human"]
        label = Q_LABELS.get(q_key, q_key).replace("\n", " ")
        print(f"{label:<30} {ai_m:>8.2f} {ai_sd:>7.2f} {h_m:>8.2f} {h_sd:>7.2f}")
        desc_rows.append({
            "Scenario": scenario_name,
            "Question": label,
            "AI_Mean": ai_m, "AI_SD": ai_sd,
            "Human_Mean": h_m, "Human_SD": h_sd,
        })

desc_df = pd.DataFrame(desc_rows)


# ==============================================================================
# SECTION 4: ASSUMPTION CHECKING
# "Shapiro-Wilk test was used to verify normality"
# "Levene's test was used to verify homogeneity of variances"
# ==============================================================================

print("\n" + "=" * 70)
print("ASSUMPTION CHECKING — SHAPIRO-WILK & LEVENE'S TEST")
print("=" * 70)
print("(Demonstrated for Scenario 1, Q4 Engagement — the significant finding)")

s1_q4 = df[(df["Scenario"] == "Scenario 1: Lecture Delivery") & (df["Question"] == "Q4_Engagement")]
ai_scores    = s1_q4[s1_q4["Instructor_Type"] == "AI"]["Score"].values
human_scores = s1_q4[s1_q4["Instructor_Type"] == "Human"]["Score"].values

sw_ai_stat,    sw_ai_p    = shapiro(ai_scores)
sw_human_stat, sw_human_p = shapiro(human_scores)
lev_stat,      lev_p      = levene(ai_scores, human_scores)

print(f"\nShapiro-Wilk — AI Scores    : W={sw_ai_stat:.4f}, p={sw_ai_p:.4f}"
      + (" ✓ Normal" if sw_ai_p > 0.05 else " ✗ Non-normal"))
print(f"Shapiro-Wilk — Human Scores : W={sw_human_stat:.4f}, p={sw_human_p:.4f}"
      + (" ✓ Normal" if sw_human_p > 0.05 else " ✗ Non-normal"))
print(f"Levene's Test (Variance)    : F={lev_stat:.4f}, p={lev_p:.4f}"
      + (" ✓ Equal variances" if lev_p > 0.05 else " ✗ Unequal variances"))
print("\nNote: As per dissertation (Section 3.5), the Mann-Whitney U test was")
print("performed to verify results. Both t-test and Mann-Whitney yielded the")
print("same conclusions, so the t-test was used as the primary test throughout.")


# ==============================================================================
# SECTION 5: INDEPENDENT SAMPLES T-TEST & MANN-WHITNEY U
# "Independent sample t-tests were used to examine mean differences"
# ==============================================================================

print("\n" + "=" * 70)
print("STATISTICAL TESTS — ALL SCENARIOS (t-test + Mann-Whitney U)")
print("=" * 70)

results_rows = []
for scenario_name, params in ALL_SCENARIOS.items():
    print(f"\n{'─'*70}")
    print(f"  {scenario_name}")
    print(f"{'─'*70}")
    print(f"  {'Question':<28} {'t-stat':>8} {'p-value':>9} {'MWU U':>8} {'MWU p':>8} {'Sig?'}")
    print(f"  {'─'*60}")
    for q_key, vals in params.items():
        ai_s    = simulate_likert(*vals["ai"])
        human_s = simulate_likert(*vals["human"])
        t_stat, t_p   = ttest_ind(ai_s, human_s)
        u_stat, mwu_p = mannwhitneyu(ai_s, human_s, alternative="two-sided")
        sig = "*** YES" if t_p < 0.05 else "no"
        label = Q_LABELS.get(q_key, q_key).replace("\n", " ")
        print(f"  {label:<28} {t_stat:>8.3f} {t_p:>9.4f} {u_stat:>8.1f} {mwu_p:>8.4f}  {sig}")
        results_rows.append({
            "Scenario": scenario_name, "Question": label,
            "AI_Mean": vals["ai"][0], "AI_SD": vals["ai"][1],
            "Human_Mean": vals["human"][0], "Human_SD": vals["human"][1],
            "t_statistic": round(t_stat, 3), "p_value": round(t_p, 4),
            "MWU_U": u_stat, "MWU_p": round(mwu_p, 4),
            "Significant": t_p < 0.05,
            "Published_t": vals["t"], "Published_p": vals["p"],
        })

results_df = pd.DataFrame(results_rows)

print(f"\n\nStatistically Significant Findings (p < 0.05):")
sig_df = results_df[results_df["Significant"]]
for _, row in sig_df.iterrows():
    print(f"  ► {row['Scenario'][:30]} | {row['Question']:<25} | t={row['t_statistic']:.3f}, p={row['p_value']:.4f}")


# ==============================================================================
# SECTION 6: VISUALISATIONS
# Reproducing the bar chart style used throughout the dissertation
# ==============================================================================

print("\n" + "=" * 70)
print("GENERATING VISUALISATIONS")
print("=" * 70)

# ── Helper: single scenario comparison bar chart ───────────────────────────────
def plot_scenario_comparison(scenario_name, params, ax, title_short):
    questions  = list(params.keys())
    ai_means   = [params[q]["ai"][0]    for q in questions]
    hum_means  = [params[q]["human"][0] for q in questions]
    ai_sds     = [params[q]["ai"][1]    for q in questions]
    hum_sds    = [params[q]["human"][1] for q in questions]
    p_vals     = [params[q]["p"]        for q in questions]
    x_labels   = [Q_LABELS.get(q, q)   for q in questions]

    x     = np.arange(len(questions))
    width = 0.35

    bars_ai  = ax.bar(x - width/2, ai_means,  width, yerr=ai_sds,  capsize=4,
                      color=COLOR_AI,    label="AI Instructor",    alpha=0.88, error_kw={"linewidth":1.2})
    bars_hum = ax.bar(x + width/2, hum_means, width, yerr=hum_sds, capsize=4,
                      color=COLOR_HUMAN, label="Human Instructor", alpha=0.88, error_kw={"linewidth":1.2})

    # Mark significant differences
    for i, p in enumerate(p_vals):
        if p < 0.05:
            ymax = max(ai_means[i] + ai_sds[i], hum_means[i] + hum_sds[i]) + 0.3
            ax.annotate("*", xy=(x[i], ymax), ha="center", fontsize=14,
                        color=COLOR_SIG, fontweight="bold")

    ax.set_xlabel("Measure", fontsize=9)
    ax.set_ylabel("Mean Score (1–7 Scale)", fontsize=9)
    ax.set_title(title_short, fontsize=10, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.set_ylim(0, 8.0)
    ax.axhline(y=4, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Figure 1: All 4 Scenarios side-by-side ────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))
fig.suptitle("AI Instructor vs. Human Instructor: Comparison Across All Scenarios\n"
             "(* = statistically significant difference, p < 0.05)",
             fontsize=13, fontweight="bold", y=1.01)

scenario_titles = [
    "Scenario 1: Lecture Delivery",
    "Scenario 2: Answering Questions",
    "Scenario 3: Providing Feedback",
    "Scenario 4: Assignment Marking",
]
axes_flat = axes.flatten()
for i, (scenario_name, params) in enumerate(ALL_SCENARIOS.items()):
    plot_scenario_comparison(scenario_name, params, axes_flat[i], scenario_titles[i])

plt.tight_layout()
plt.savefig("figure1_all_scenarios_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: figure1_all_scenarios_comparison.png")


# ── Figure 2: Engagement — the key significant finding (Scenario 1, Q4) ───────
fig, ax = plt.subplots(figsize=(7, 5))
categories  = ["AI Instructor", "Human Instructor"]
means       = [4.76, 5.33]
sds         = [1.47, 1.23]
colors      = [COLOR_AI, COLOR_HUMAN]

bars = ax.bar(categories, means, yerr=sds, capsize=6,
              color=colors, alpha=0.88, width=0.45,
              error_kw={"linewidth": 1.5, "ecolor": "dimgray"})
ax.set_ylabel("Mean Engagement Score (1–7 Scale)", fontsize=11)
ax.set_title("Perceived Engagement: AI vs Human Instructor-Led Lectures\n"
             "(Scenario 1, Q4 — p = 0.030, statistically significant)",
             fontsize=10, fontweight="bold")
ax.set_ylim(0, 7.5)
ax.axhline(y=4, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Midpoint")
ax.annotate("p = 0.030 *", xy=(0.5, 6.3), xycoords="data",
            ha="center", fontsize=11, color=COLOR_SIG, fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("figure2_engagement_significant.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: figure2_engagement_significant.png")


# ── Figure 3: Mean Scores Heatmap across all questions ────────────────────────
pivot_data = {}
for scenario_name, params in ALL_SCENARIOS.items():
    for q_key, vals in params.items():
        label = Q_LABELS.get(q_key, q_key).replace("\n", " ")
        key = scenario_name.split(":")[1].strip() + " | " + label
        pivot_data[key] = {"AI": vals["ai"][0], "Human": vals["human"][0], "Diff": vals["human"][0] - vals["ai"][0]}

heatmap_df = pd.DataFrame(pivot_data).T
fig, ax = plt.subplots(figsize=(6, 12))
sns.heatmap(heatmap_df[["AI", "Human", "Diff"]].astype(float),
            annot=True, fmt=".2f", cmap="RdYlGn", ax=ax,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Mean Score / Difference"},
            vmin=0, vmax=7)
ax.set_title("Mean Scores: AI vs Human Instructor\n(All Scenarios & Questions)",
             fontsize=11, fontweight="bold", pad=10)
ax.set_xticklabels(["AI Mean", "Human Mean", "Difference (H-A)"], fontsize=10)
ax.tick_params(axis="y", labelsize=8)
plt.tight_layout()
plt.savefig("figure3_heatmap_all_scores.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: figure3_heatmap_all_scores.png")


# ── Figure 4: p-value comparison across all tests ─────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
labels  = [f"S{i//6+1}-Q{i%6+1}" for i in range(24)]
p_vals  = []
for params in ALL_SCENARIOS.values():
    for q_key, vals in params.items():
        p_vals.append(vals["p"])

bar_colors = [COLOR_SIG if p < 0.05 else COLOR_AI for p in p_vals]
bars = ax.bar(labels, p_vals, color=bar_colors, alpha=0.85, edgecolor="white")
ax.axhline(y=0.05, color="red", linestyle="--", linewidth=1.5,
           label="Significance threshold (p = 0.05)")
ax.set_ylabel("p-value", fontsize=11)
ax.set_xlabel("Test (Scenario-Question)", fontsize=11)
ax.set_title("p-values Across All Statistical Tests\n(Red bars = statistically significant, p < 0.05)",
             fontsize=11, fontweight="bold")
ax.set_ylim(0, 0.75)
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

sig_patch   = mpatches.Patch(color=COLOR_SIG, alpha=0.85, label="Significant (p < 0.05)")
nosig_patch = mpatches.Patch(color=COLOR_AI,  alpha=0.85, label="Not significant")
ax.legend(handles=[sig_patch, nosig_patch,
                   plt.Line2D([0],[0], color="red", linestyle="--", linewidth=1.5,
                              label="Threshold (p=0.05)")],
          fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig("figure4_pvalue_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: figure4_pvalue_overview.png")


# ── Figure 5: Overall AI vs Human mean by scenario ────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
scenario_labels = ["Lecture\nDelivery", "Answering\nQuestions", "Providing\nFeedback", "Assignment\nMarking"]
overall_ai_means    = []
overall_human_means = []
for params in ALL_SCENARIOS.values():
    overall_ai_means.append(np.mean([v["ai"][0]    for v in params.values()]))
    overall_human_means.append(np.mean([v["human"][0] for v in params.values()]))

x     = np.arange(4)
width = 0.35
ax.bar(x - width/2, overall_ai_means,    width, color=COLOR_AI,    label="AI Instructor",    alpha=0.88)
ax.bar(x + width/2, overall_human_means, width, color=COLOR_HUMAN, label="Human Instructor", alpha=0.88)
ax.set_xticks(x)
ax.set_xticklabels(scenario_labels, fontsize=10)
ax.set_ylabel("Overall Mean Score (1–7)", fontsize=11)
ax.set_ylim(0, 7)
ax.set_title("Overall Mean Scores: AI vs Human Instructor by Scenario",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("figure5_overall_scenario_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✓ Saved: figure5_overall_scenario_comparison.png")


# ==============================================================================
# SECTION 7: RESULTS SUMMARY TABLE
# ==============================================================================

print("\n" + "=" * 70)
print("RESULTS SUMMARY — PUBLISHED vs REPRODUCED")
print("=" * 70)
print(f"\n{'Scenario':<35} {'Question':<25} {'Pub t':>7} {'Pub p':>8} {'Sig?':>6}")
print("-" * 82)
for scenario_name, params in ALL_SCENARIOS.items():
    s_label = scenario_name.split(":")[1].strip()
    for q_key, vals in params.items():
        q_label = Q_LABELS.get(q_key, q_key).replace("\n", " ")
        sig = "YES *" if vals["p"] < 0.05 else "no"
        print(f"{s_label:<35} {q_label:<25} {vals['t']:>7.2f} {vals['p']:>8.4f} {sig:>6}")


# ==============================================================================
# SECTION 8: KEY FINDINGS SUMMARY
# ==============================================================================

print("\n" + "=" * 70)
print("KEY FINDINGS — DISSERTATION SUMMARY")
print("=" * 70)

findings = [
    ("Scenario 1 — Lecture Delivery",
     "Human instructors significantly more ENGAGING (p=0.030). No significant\n"
     "  difference in effectiveness, accuracy, competency, or likelihood of attendance."),
    ("Scenario 2 — Answering Questions",
     "Human instructors significantly more EFFECTIVE (p=0.018) and generated\n"
     "  more positive INITIAL THOUGHTS (p=0.0185). Other measures not significant."),
    ("Scenario 3 — Providing Feedback",
     "No statistically significant differences across all 6 measures.\n"
     "  AI feedback regarded as near-equivalent to human feedback."),
    ("Scenario 4 — Assignment Marking",
     "No statistically significant differences across all 6 measures.\n"
     "  AI considered a credible substitute for routine marking tasks."),
    ("Overall Conclusion",
     "AI instructors competent for standardised tasks (marking, feedback).\n"
     "  Human instructors preferred for engagement and interaction.\n"
     "  A hybrid AI-human model recommended for higher education."),
]

for heading, text in findings:
    print(f"\n  ► {heading}")
    print(f"    {text}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("Files generated: figure1–5 PNG charts + this script")
print("=" * 70)
