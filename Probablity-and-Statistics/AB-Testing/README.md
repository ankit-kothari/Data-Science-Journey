# A/B Testing for Recommendations & Personalization

End-to-end reference for designing, running, and analyzing experiments on recommendation systems and personalization features.

For a worked example, see the [Cookie Cats retention case study](ab-testing-cookie-cat-dataset.ipynb).

---

**Contents:**
[1. Experiment Design](#1-experiment-design) · [2. Hypothesis Formulation](#2-hypothesis-formulation) · [3. Significance & Power](#3-significance-and-power) · [4. Metric Selection](#4-metric-selection) · [5. Effect Size](#5-effect-size) · [6. Sample Size](#6-sample-size-calculation) · [7. Distributions](#7-data-distributions) · [8. Parametric Tests](#8-parametric-tests) · [9. Non-Parametric Tests](#9-non-parametric-tests) · [10. Drawing Conclusions](#10-drawing-conclusions) · [11. Pros & Cons](#11-pros-and-cons) · [12. Resources](#12-resources)

---

## 1. Experiment Design

An A/B test measures whether a change (new ranking model, different carousel layout, personalized vs generic homepage) causes a meaningful shift in a target metric (CTR, conversion, retention, revenue per user).

| Group | Role | Rec-sys example |
| --- | --- | --- |
| **Control** | Existing experience | Collaborative-filtering ranker |
| **Treatment** | New variant | Two-tower retrieval + neural ranker |

Users are assigned via **random sampling** — simple, stratified (by region/platform), or cluster (by household) depending on the surface.

### Reliability techniques

| Technique | What it does | Rec-sys example |
| --- | --- | --- |
| **Blind** | Users don't know their group | All rec experiments — users never see group labels |
| **Double-blind** | Analysts also don't know during collection | Prevents bias in manual quality audits of recommendations |
| **Blocking** | Split on a known confounder before randomization | Block on platform (iOS/Android) — ranking model may perform differently per device |
| **Matched pairs** | Pair treatment/control users on shared traits | Match on tenure + purchase frequency, so heavy buyers are equally represented in both groups |

---

## 2. Hypothesis Formulation

| | Definition | Rec-sys example |
| --- | --- | --- |
| **H₀** | No difference between groups | The new two-tower ranker produces the same CTR as collaborative filtering |
| **H₁** | There is a difference | The new ranker changes CTR (up or down) |

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/FB43C87D-FC14-4F57-B2EB-149DEF1B5233.jpeg" width="40%">

### One-tailed vs two-tailed

| Test type | H₀ | H₁ | When to use |
| --- | --- | --- | --- |
| **Two-tailed** | New ranker CTR = old ranker CTR | New ranker CTR ≠ old ranker CTR | Default — you want to detect harm too |
| **Upper-tail** | New ranker CTR ≤ old ranker CTR | New ranker CTR > old ranker CTR | Only care about improvement (rare in practice) |
| **Lower-tail** | New ranker CTR ≥ old ranker CTR | New ranker CTR < old ranker CTR | Guardrail check — is the new model degrading? |

```python
from scipy.stats import mannwhitneyu

# Example: comparing revenue-per-session between control and treatment
stat, p = mannwhitneyu(control_revenue, treatment_revenue, alternative="two-sided")   # two-tailed
stat, p = mannwhitneyu(control_revenue, treatment_revenue, alternative="greater")     # upper-tail
stat, p = mannwhitneyu(control_revenue, treatment_revenue, alternative="less")        # lower-tail
```

---

## 3. Significance and Power

These control your error rates and determine how many users you need.

### Significance level (α) — Type I error rate

Probability of rejecting H₀ when H₀ is true (false positive: you ship a model that isn't actually better).

| α | Confidence | When to use |
| --- | --- | --- |
| 0.10 | 90% | Low-risk: carousel reordering, copy change |
| 0.05 | 95% | Standard: new ranking model, recommendation algorithm |
| 0.01 | 99% | High-risk: pricing personalization, checkout flow changes |

### Statistical power (1 − β) — detecting real effects

Probability of correctly rejecting H₀ when H₁ is true. β is the Type II error rate (false negative: a genuinely better model gets killed).

| 1 − β | When to use |
| --- | --- |
| 0.80 | Standard — most rec-sys experiments |
| 0.90 | High-value surfaces (homepage, search ranking) |
| 0.95+ | Revenue-critical personalization (pricing, promotions) |

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/4F78D489-6828-4925-A1FF-26FFD92289AF.jpeg" width="40%">

### The four-way relationship

**α**, **power**, **effect size**, and **sample size** are mathematically linked. Fix any three → the fourth is determined. In practice: fix α + power + minimum detectable effect → solve for sample size.

---

## 4. Metric Selection

### Invariant metrics (sanity checks)

Should **not** change between groups. If they do, randomization is broken.

| Invariant metric | What it catches |
| --- | --- |
| Users per group | Uneven traffic splitting |
| Platform distribution (iOS/Android) | Sampling bias |
| Avg page load time | Treatment causing latency, confounding engagement |
| Recommendation request volume | Backend routing errors |

### Evaluation metrics

| Type | Rec-sys examples | Typical effect size | Sample implication |
| --- | --- | --- | --- |
| **Binary** | Purchase conversion, add-to-cart, click-through | 0.1–0.5 pp absolute | Needs 10–50K users/group |
| **Continuous** | Revenue/session, units/order, watch time, session duration | d = 0.05–0.15 | Needs 1–5K users/group |

A good experiment tracks **one primary metric** plus **guardrail metrics** to catch tradeoffs:

| Primary metric | Guardrail metrics | Why |
| --- | --- | --- |
| Purchase conversion | Revenue/session, return rate, category diversity | New ranker might boost cheap impulse buys but hurt AOV |
| Units per order | Revenue/order, cart abandonment rate | Cross-sell might add low-margin items or overwhelm users |
| CTR on carousel | Conversion, session depth, rec diversity | High CTR from clickbait titles hurts downstream metrics |
| Revenue per session | Conversion rate, items viewed, latency p99 | Revenue can rise from fewer high-ticket sales while conversion drops |

---

## 5. Effect Size

Effect size = **how many standard deviations** the treatment mean (or proportion) is from control. A test can be *statistically significant* with a tiny, useless effect — effect size tells you whether the lift is *worth shipping*.

### Why effect size drives experiment planning

Three numbers you must know before running any experiment: **baseline value**, **minimum detectable effect (MDE)**, and **standard deviation**. These determine whether you can even *detect* the lift you care about given your traffic — and whether the lift is worth pursuing. Reasoning about practical significance, not just p-values, is what separates good experiment design from naive hypothesis testing.

### Baseline metric — anchor every experiment

Establish the current value *and its variance* before running the test.

> **Scenario — conversion rate:** Your e-commerce "Recommended for You" carousel has a **3.2% purchase conversion rate** (baseline = 0.032, σ ≈ 0.176). A new two-tower retrieval model should increase this. With a 3.2% baseline, even a 0.3 pp absolute lift (3.2% → 3.5%) is a **+9.4% relative lift** — meaningful for revenue. But if your baseline were 40%, a 0.3 pp lift is noise.
>
> **Scenario — units per order:** Your "Complete the Look" cross-sell widget shows avg 2.3 units per order (σ = 1.8). A personalized version targets 2.5 units. The absolute lift is 0.2 units, but you need to know σ to judge whether that's detectable.

**Rule of thumb:** low-baseline metrics (conversion, add-to-cart) need large *relative* lifts to be detectable. High-baseline metrics (CTR on a carousel) can detect smaller relative changes.

### Effect size for means — Cohen's d

`d = (μ_treatment − μ_control) / σ_pooled`

| d | Interpretation | Rec-sys example |
| --- | --- | --- |
| 0.02–0.05 | Very small | Units per order: 2.3 → 2.4 (σ = 1.8) → d = 0.056 |
| 0.10–0.20 | Small | Revenue/session: $8.50 → $9.20 (σ = $6.00) → d = 0.117 |
| 0.50 | Medium | Watch time: 25 min → 32 min (σ = 14 min) → d = 0.50 |
| 0.80+ | Large | Rare in production rec-sys experiments |

```python
# Scenario 1: personalized cross-sell widget — does it increase units per order?
# Baseline: mean = 2.3 units, std = 1.8, target = 2.5
d = (2.5 - 2.3) / 1.8  # d ≈ 0.11 — small effect, needs ~1,300 users/group

# Scenario 2: new ranking model — does it increase revenue per session?
# Baseline: mean = $8.50, std = $6.00, target = $9.20
d = (9.20 - 8.50) / 6.00  # d ≈ 0.117 — similar magnitude

# Scenario 3: homepage feed rerank — does it increase session duration?
# Baseline: mean = 12.4 min, std = 8.1 min, target = 13.0 min
d = (13.0 - 12.4) / 8.1  # d ≈ 0.074 — very small, needs ~2,900 users/group
```

**In practice:** most production rec-sys experiments have d between 0.02 and 0.15. This is why you need thousands of users — the effects are real but small relative to user-level variance.

### Effect size for proportions — Cohen's h

`h = 2 × (arcsin(√p₂) − arcsin(√p₁))`

The arcsin transform stabilizes variance across different baseline rates — a 1 pp lift from 3% → 4% is a bigger *relative* signal than 50% → 51%.

```python
import statsmodels.stats.api as sms

# Scenario 1: purchase conversion — new retrieval model
# Baseline: 3.2% conversion, want to detect a +10% relative lift (3.2% → 3.52%)
baseline = 0.032
target = 0.032 * 1.10  # = 0.0352
h = sms.proportion_effectsize(baseline, target)
# h ≈ 0.019 — very small; needs ~43,000 users/group at 80% power

# Scenario 2: add-to-cart rate — personalized "You might also like"
# Baseline: 8.5% add-to-cart, want to detect +2 pp absolute lift (8.5% → 10.5%)
baseline = 0.085
target = 0.105
h = sms.proportion_effectsize(baseline, target)
# h ≈ 0.068 — still small; needs ~3,400 users/group

# Scenario 3: carousel CTR — reranking by user affinity
# Baseline: 15% CTR, want to detect +3 pp absolute lift (15% → 18%)
baseline = 0.15
target = 0.18
h = sms.proportion_effectsize(baseline, target)
# h ≈ 0.082 — moderate; needs ~2,300 users/group
```

**In practice:** conversion rates have small absolute values (3–8%), so even meaningful business lifts produce tiny effect sizes. This is why conversion experiments run longer than engagement experiments — and why many teams use **revenue per user** (continuous, higher variance but also higher signal) as the primary metric instead.

### Choosing the MDE — the business question behind the math

| Metric | Typical baseline | Realistic MDE | Why |
| --- | --- | --- | --- |
| Purchase conversion | 2–5% | 5–15% relative | Below 5% relative lift, revenue gain doesn't justify eng cost |
| Units per order | 2.0–3.5 | 0.1–0.3 units | Each +0.1 unit ≈ +$3–5 AOV at scale |
| Revenue per session | $5–15 | 3–8% relative | Directly tied to topline; small % = large $ at volume |
| CTR (carousel/feed) | 10–20% | 1–3 pp absolute | CTR is a proxy — only matters if downstream conversion follows |
| Session duration | 8–20 min | 5–10% relative | Engagement proxy; diminishing returns past some threshold |

---

## 6. Sample Size Calculation

### For proportions (binary: converted / didn't convert)

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.20.38_PM.png" width="40%">

Where **ω** = minimum detectable change.

```python
import statsmodels.stats.api as sms

# Scenario: new two-tower retrieval model — does it lift purchase conversion?
# Baseline conversion = 3.2%, want to detect +10% relative lift (→ 3.52%)
baseline = 0.032
target = 0.032 * 1.10  # 3.52% — a 0.32 pp absolute lift
alpha = 0.05
power = 0.80

effect_size = sms.proportion_effectsize(baseline, target)
n = sms.NormalIndPower().solve_power(
    effect_size=effect_size, power=power, alpha=alpha, ratio=1
)
print(f"Required: {n:.0f} users per group")
# Output: ~43,000 users per group
# At 100K daily visitors (50/50 split): ~17 days to reach sample size

# Scenario 2: add-to-cart rate for personalized cross-sell
# Baseline = 8.5%, want to detect +2 pp absolute lift (→ 10.5%)
effect_size_2 = sms.proportion_effectsize(0.085, 0.105)
n2 = sms.NormalIndPower().solve_power(
    effect_size=effect_size_2, power=0.80, alpha=0.05, ratio=1
)
print(f"Required: {n2:.0f} users per group")
# Output: ~3,400 users per group — much faster because the absolute lift is larger
```

### For means (continuous: revenue per session, units per order)

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.25.59_PM.png" width="40%">

```python
import statsmodels.stats.api as sms

# Scenario: personalized cross-sell — does it increase units per order?
# Baseline: mean = 2.3 units, std = 1.8, target = 2.5 → d = 0.11
d = (2.5 - 2.3) / 1.8
n = sms.TTestIndPower().solve_power(effect_size=d, power=0.80, alpha=0.05)
print(f"Units per order: {n:.0f} users per group")
# Output: ~1,300 users per group

# Scenario 2: new ranking model — revenue per session
# Baseline: mean = $8.50, std = $6.00, target = $9.20 → d = 0.117
d2 = (9.20 - 8.50) / 6.00
n2 = sms.TTestIndPower().solve_power(effect_size=d2, power=0.80, alpha=0.05)
print(f"Revenue/session: {n2:.0f} users per group")
# Output: ~1,150 users per group — continuous metrics need fewer users than proportions
```

### Trade-off: sample size vs detectable effect

Smaller effects need exponentially more users. This is the core planning decision: is detecting a 0.3 pp conversion lift worth running the test for 3 weeks instead of 1?

| Metric | MDE | Users/group | Days at 50K/day (50/50) |
| --- | --- | --- | --- |
| Conversion (3.2%) | +1.0 pp | ~4,200 | < 1 day |
| Conversion (3.2%) | +0.3 pp | ~43,000 | ~2 days |
| Conversion (3.2%) | +0.1 pp | ~390,000 | ~16 days |
| Units/order (2.3, σ=1.8) | +0.2 units | ~1,300 | < 1 day |
| Units/order (2.3, σ=1.8) | +0.05 units | ~20,500 | < 1 day |

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/effect.png" width="40%">

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms

# How sample size explodes as you try to detect smaller conversion lifts
baseline = 0.032  # 3.2% purchase conversion
deltas = np.arange(0.001, 0.015, 0.0005)  # 0.1 pp to 1.5 pp absolute lifts
sizes = []

for delta in deltas:
    es = sms.proportion_effectsize(baseline, baseline + delta)
    n = sms.NormalIndPower().solve_power(effect_size=es, power=0.8, alpha=0.05, ratio=1)
    sizes.append(n)

plt.plot(deltas * 100, sizes)
plt.axhline(y=50_000, color="r", linestyle="--", label="50K users (typical 2-day budget)")
plt.title("Sample Size vs Minimum Detectable Conversion Lift (baseline=3.2%)")
plt.ylabel("Users per Group")
plt.xlabel("Minimum Detectable Lift (percentage points)")
plt.legend()
plt.tight_layout()
plt.show()
```

### Running the experiment

- **Don't stop early.** Peeking inflates false positive rates. Commit to the calculated sample size.
- **Traffic split** doesn't have to be 50/50. Allocate 90/10 if the new model is risky (e.g., untested ranker on high-traffic surface), but expect longer test duration.

### Pre-analysis checklist

1. **Invariant metrics** — confirm no significant difference. If users-per-group or platform-split is off, the experiment is compromised.
2. **Distribution check** — plot density/boxplots for each evaluation metric per group. Run **Shapiro-Wilk** to test normality.
3. **Choose test** — normal data → parametric; non-normal or binary → non-parametric.
4. **Report correctly** — normal: mean ± CI. Non-normal: median with Q1/Q3.

---

## 7. Data Distributions

### Binomial distribution

**Rec-sys use case:** Did the user click the recommendation? (yes/no). Did the user convert? (yes/no).

Conditions: (1) independent trials, (2) two outcomes, (3) fixed n, (4) constant probability.

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.42.09_PM.png" width="40%">

**PMF:** P(x successes in n trials)

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_12.38.15_PM.png" width="40%">

```python
from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

# Scenario: comparing click counts on control vs treatment carousel
clicks = np.arange(20, 80)
n_control, n_treatment = 550, 450
clicked_control, clicked_treatment = 48, 56
rate_control = clicked_control / n_control   # 8.7% CTR
rate_treatment = clicked_treatment / n_treatment  # 12.4% CTR

prob_control = binom(n_control, rate_control).pmf(clicks)
prob_treatment = binom(n_treatment, rate_treatment).pmf(clicks)

plt.bar(clicks, prob_control, label="Control (CF ranker)", alpha=0.7)
plt.bar(clicks, prob_treatment, label="Treatment (two-tower)", alpha=0.7)
plt.legend()
plt.xlabel("Number of clicks")
plt.ylabel("Probability")
plt.show()
```

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/binomial.png" width="40%">

### Normal distribution

**Rec-sys use case:** Revenue per session, watch time, time-to-first-click — continuous metrics that tend toward normal with enough users.

Key properties:

| Property | Value |
| --- | --- |
| Mean = median = mode | Center of distribution |
| 1σ | 68% of data |
| 2σ | 95% of data |
| 3σ | 99.7% of data |

**PDF:**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_9.57.05_PM.png" width="40%">

PDF gives density at a point — how concentrated probability is near x. Actual probability over an interval = integral of PDF.

**Effect of shifting mean** (σ fixed) — curve slides horizontally:

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-19_at_2.04.23_AM.png" width="40%">

**Effect of increasing σ** (mean fixed) — curve flattens and widens:

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-19_at_1.58.49_AM.png" width="40%">

#### Z-score and z-statistic

**Z-score** = distance from the mean in standard deviations: `z = (X − μ) / σ`

> Example: avg session revenue μ = $4.20, σ = $1.00. A user with $5.70 has z = 1.5 → z-table says 93.38% of users spend less. This area is the p-value for a one-tailed test.

**When to use z vs t:**

| Condition | Statistic |
| --- | --- |
| Population σ known | z |
| σ unknown, n > 30 | z (sample σ ≈ population σ) |
| σ unknown, n ≤ 30 | t |

**z-statistic formulas:**

One-sample:

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/9EFFB6E2-024B-4DBB-B5DE-9065943FF75D.jpeg" width="40%">

Sample (σ unknown, n > 30):

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/67898C9F-77A8-4CA7-B993-73D0E8B21F36.jpeg" width="40%">

Two proportions (e.g., comparing CTR between groups):

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-18_at_10.37.00_AM.png" width="40%">

```python
from statsmodels.stats.proportion import proportions_ztest
import numpy as np

# 486 out of 5000 clicked in control, 527 out of 5000 in treatment
conversions = np.array([486, 527])
users = np.array([5000, 5000])

zscore, pvalue = proportions_ztest(conversions, users, alternative="two-sided")
print(f"z = {zscore:.4f}, p = {pvalue:.4f}")
# z = -1.3589, p = 0.1742 → fail to reject H₀, no significant CTR difference
```

### Student's t-distribution

Used when n is small (< 30). Shape depends on **degrees of freedom (df = n − 1)**: low df → heavier tails. Converges to normal as df increases.

**Rec-sys use case:** Pilot test of a new recommendation model on a small user cohort (e.g., 25 users per group).

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.11.25_PM.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.10.57_PM.png" width="40%">

**t-distribution PDF:**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.11.15_PM.png" width="40%">

**t-score:**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_10.11.31_PM.png" width="40%">

| Condition | Statistic |
| --- | --- |
| σ unknown, n > 30 | t (also valid with z) |
| σ unknown, n ≤ 30 | t (requires approx. normal population) |

### Chi-square distribution

Special case of gamma distribution. One parameter: **degrees of freedom (ν)**.

**Rec-sys use case:** Do users in control vs treatment groups distribute differently across product categories (electronics, clothing, books)?

Properties: positive values only, right-skewed. Mean = ν, std = √(2ν). Approaches normal as ν → ∞.

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.30.37_PM.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.30.41_PM.png" width="40%">

**Chi-square statistic:** χ² = Σ (Oᵢ − Eᵢ)² / Eᵢ

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.33.19_PM.png" width="40%">

**PDF** (k = degrees of freedom):

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-16_at_11.35.38_PM.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/Screen_Shot_2020-08-20_at_5.33.06_PM.png" width="40%">

---

## 8. Parametric Tests

Assume data is normally distributed. Use for continuous metrics like revenue, session time, items viewed.

### Z-test

**When:** Population σ known, or n > 30.

```python
import numpy as np
from scipy.stats import norm

def ztest_two_proportions(X1: int, X2: int, n1: int, n2: int):
    """Compare conversion rates between control and treatment.

    X1, X2: conversions (clicks, purchases) per group
    n1, n2: total users per group
    """
    p1_hat = X1 / n1
    p2_hat = X2 / n2
    p_bar = (X1 + X2) / (n1 + n2)
    q_bar = 1 - p_bar

    se = np.sqrt((1/n1 + 1/n2) * p_bar * q_bar)
    z_score = (p1_hat - p2_hat) / se
    p_value = norm.cdf(z_score)  # one-tailed; × 2 for two-tailed

    return z_score, p_value, se
```

### T-test

**When:** Population σ unknown. Assumes normal distribution.

| Variant | Rec-sys use case |
| --- | --- |
| **Independent samples** | Units per order: personalized cross-sell vs rule-based |
| **Paired samples** | Same users, before/after a ranking model change |
| **One-sample** | Is avg revenue/session in treatment > $8.50 baseline? |

```python
from scipy.stats import ttest_ind
import numpy as np

# Scenario: personalized cross-sell widget — does it increase units per order?
# Control: rule-based "Customers also bought" (mean=2.3, std=1.8)
# Treatment: personalized widget using purchase history (mean=2.5, std=1.9)
units_control = np.random.normal(2.3, 1.8, 5000)
units_treatment = np.random.normal(2.5, 1.9, 5000)

t_score, p_value = ttest_ind(units_control, units_treatment, equal_var=True)
print(f"t = {t_score:.2f}, p = {p_value:.4f} (two-tailed)")
```

### Welch's t-test

**When:** Normal data, but **unequal sample sizes and/or unequal variances** — the default in practice. Always prefer Welch's over standard t-test unless you've confirmed equal variance.

```python
from scipy.stats import ttest_ind
import numpy as np

# Unequal groups: 70/30 traffic split to reduce risk of new ranker
# Revenue per session — higher variance metric, unequal group sizes
revenue_control = np.random.normal(8.50, 6.00, 7000)
revenue_treatment = np.random.normal(9.20, 6.50, 3000)

t_score, p_value = ttest_ind(revenue_control, revenue_treatment, equal_var=False)
print(f"t = {t_score:.2f}, p = {p_value:.4f} (two-tailed, Welch)")
```

**Why Welch's matters:** Standard t-test with unequal variance inflates false positive rates — the rejection area in [0, 0.05] is much larger than the nominal 5%.

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ttest.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/welch_test.png" width="40%">

For binary data (clicked/didn't), use Mann-Whitney U instead.

### ANOVA (F-statistic)

**When:** Comparing **3+ groups** — e.g., testing three different ranking models simultaneously.

| Variant | Use case |
| --- | --- |
| **One-way ANOVA** | One factor (ranker variant), multiple groups |
| **MANOVA** | Multiple factors (ranker + layout), multiple metrics (CTR + revenue) |

---

## 9. Non-Parametric Tests

No distribution assumption. Use for binary outcomes (click/no-click, convert/churn) or non-normal continuous data.

### Chi-squared test

**When:** Categorical outcomes with 2+ categories.

**Rec-sys use case:** Does the new ranker change the distribution of which product categories users click on?

Two variants:
- **Goodness of fit** — does the category distribution match the expected baseline?
- **Test of independence** — is conversion rate independent of which ranker group users are in?

**Method 1: Proportions test**

```python
import statsmodels.stats.proportion as proportion
import numpy as np

# 486/5000 converted in control, 527/5000 in treatment
converted = np.array([486, 527])
users = np.array([5000, 5000])

chisq, pvalue, table = proportion.proportions_chisquare(converted, users)
print(f"χ² = {chisq:.3f}, p = {pvalue:.3f}")
```

**Method 2: Contingency table**

```python
import scipy.stats as stats
import numpy as np

# rows: [didn't convert, converted] per group
observed = np.array([[4514, 486], [4473, 527]])

chisq, pvalue, dof, expected = stats.chi2_contingency(observed, correction=False)
print(f"χ² = {chisq:.3f}, p = {pvalue:.3f}")
```

### Fisher's exact test

**When:** Small sample or imbalanced categories where chi-squared approximation breaks down.

**Rec-sys use case:** Testing a new model on a niche category with few conversions (e.g., luxury items — 50 purchases across both groups).

```python
from scipy.stats import fisher_exact

# [purchased, didn't purchase] for control vs treatment
oddsratio, pvalue = fisher_exact([[50, 2450], [42, 2458]])
print(f"odds ratio = {oddsratio:.3f}, p = {pvalue:.4f}")
```

### Mann-Whitney U test

**When:** Two independent groups, non-normal distribution. Non-parametric alternative to the independent t-test.

**Rec-sys use case:** Comparing number of items added to cart (highly skewed — most users add 0, some add 20+).

Requirements: n > 20 per group. Compares rank distributions — works on any shape.

```python
from scipy.stats import mannwhitneyu

# Items added to cart per session (skewed distribution)
cart_control = [0, 0, 0, 1, 0, 2, 0, 0, 3, 0, 1, 0, 0, 0, 5, 0, 0, 1, 0, 0, 0]
cart_treatment = [0, 1, 0, 2, 0, 0, 3, 1, 0, 0, 0, 4, 0, 1, 0, 2, 0, 0, 1, 0, 3]

stat, pvalue = mannwhitneyu(cart_control, cart_treatment)
print(f"U = {stat:.1f}, p = {pvalue:.4f}")

if pvalue > 0.05:
    print("Fail to reject H₀: no significant difference in cart additions")
else:
    print("Reject H₀: treatment changed cart behavior")
```

### Wilcoxon signed-rank test

**When:** Paired samples, non-normal data. Non-parametric alternative to the paired t-test.

**Rec-sys use case:** Same users exposed to old and new ranker in sequential periods — compare the per-user difference in engagement.

Assumptions: random representative samples, independent, values have natural order.

```python
from scipy.stats import wilcoxon

# Per-user difference in clicks/week: (new ranker period) - (old ranker period)
click_diffs = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

w, p = wilcoxon(click_diffs)
print(f"W = {w}, p = {p:.4f}")
# W = 24.0, p = 0.0413 → reject H₀ at α=0.05
```

### Quick reference: which test to use

| Data | Normal? | Groups | Test | Rec-sys example |
| --- | --- | --- | --- | --- |
| Continuous | Yes | 2, equal var | T-test | Revenue/session, equal traffic split |
| Continuous | Yes | 2, unequal var/size | **Welch's** | Revenue/session, 70/30 split |
| Continuous | Yes | 2, paired | Paired t-test | Same users, before/after |
| Continuous | Yes | 3+ | ANOVA | Testing 3 ranker variants |
| Continuous | No | 2, independent | Mann-Whitney U | Cart size (skewed) |
| Continuous | No | 2, paired | Wilcoxon | Per-user engagement diff |
| Binary | — | 2+ | Chi-squared | Conversion rate |
| Binary | — | 2, small n | Fisher's exact | Niche category conversions |
| Proportions | — | 2 | Z-test | CTR comparison |

---

## 10. Drawing Conclusions

A result must clear **two bars** to ship:

### Bar 1: Statistically significant

p-value < α → the observed difference is unlikely under H₀.

### Bar 2: Practically significant

The confidence interval lower bound exceeds the **minimum detectable effect** you defined upfront. A result can be statistically significant but too small to justify the cost of shipping (model training infra, latency overhead, eng maintenance).

> **Scenario — ship:** You tested a personalized cross-sell widget against a rule-based "Customers also bought" widget. MDE was +0.15 units per order. Result: mean lift = +0.24 units, 95% CI = [+0.17, +0.31]. The CI lower bound (0.17) exceeds the MDE (0.15) → **ship it**. At 500K orders/month, +0.24 units ≈ +120K incremental items/month.
>
> **Scenario — don't ship:** You tested a new two-tower ranker for homepage recs. MDE was +0.3 pp conversion. Result: mean lift = +0.12 pp, 95% CI = [−0.05 pp, +0.29 pp]. The CI includes zero *and* the upper bound doesn't reach 0.3 pp → **keep the existing model**. The new ranker adds 40ms latency and 2 eng-months of maintenance — not worth a lift you can't even confirm exists.
>
> **Scenario — significant but not practical:** The same ranker test with 10× more traffic shows lift = +0.08 pp, 95% CI = [+0.02 pp, +0.14 pp]. Now p < 0.05 (CI excludes zero), but the upper bound (0.14 pp) is still below your 0.3 pp MDE. **Statistically significant, practically useless** — this is the most common mistake in experiment readouts.

### Computing the confidence interval

```python
import math
import scipy.stats as st

# Worked example: new two-tower ranker for purchase conversion
# Control: 1,600 conversions out of 50,000 users (3.20%)
# Treatment: 1,660 conversions out of 50,000 users (3.32%)
n_ctrl, n_treat = 50_000, 50_000
conv_ctrl, conv_treat = 1_600, 1_660
alpha = 0.05
mde = 0.003  # 0.3 pp — our minimum detectable effect

p_ctrl = conv_ctrl / n_ctrl        # 0.0320
p_treat = conv_treat / n_treat      # 0.0332
p_pooled = (conv_ctrl + conv_treat) / (n_ctrl + n_treat)  # 0.0326

se = math.sqrt(p_pooled * (1 - p_pooled) * (1/n_ctrl + 1/n_treat))
z = st.norm.ppf(1 - alpha / 2)     # 1.96
margin = se * z

d_hat = p_treat - p_ctrl            # +0.0012 (0.12 pp)
lower = d_hat - margin
upper = d_hat + margin

print(f"Conversion lift: {d_hat*100:.2f} pp")
print(f"95% CI: [{lower*100:.2f} pp, {upper*100:.2f} pp]")
# Output: lift = 0.12 pp, CI = [-0.05 pp, +0.29 pp]

# Decision logic
if lower > 0 and lower > mde:
    print("Ship — statistically AND practically significant")
elif lower > 0:
    print("Significant but below MDE — probably not worth the eng cost")
else:
    print("CI includes zero — cannot reject H₀, keep current model")
# Output: CI includes zero — keep current model
```

---

## 11. Pros and Cons

| Pros | Cons |
| --- | --- |
| Works with low traffic — bandits/multivariate may not converge | One variable at a time — testing 5 ranker changes takes 5 sequential tests |
| Simple to implement — most experimentation platforms support it | Inefficient data collection — learnings from test 1 don't reduce sample needs for test 2 |
| Simple to design — just split traffic, no factorial design | |
| Simple to analyze — standard statistical tests | |
| Flexible — can compare evolutionary tweak vs radical redesign in one test | |

---

## 12. Resources

- [Nonparametric Statistical Significance Tests in Python](https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/)
- [A/B Testing in Real Life](https://towardsdatascience.com/ab-testing-in-real-life-9b490b3c50d1)
- [Introduction to Statistics](https://towardsdatascience.com/introduction-to-statistics-e9d72d818745)
- [A/B Tests Tutorial (Cosmic Coding)](https://cosmiccoding.com.au/tutorials/ab_tests)
- [Ultimate Guide: Parametric Tests](https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-3-parametric-tests-2c629e8d98f8)
- [Ultimate Guide: Non-Parametric Tests](https://towardsdatascience.com/the-ultimate-guide-to-a-b-testing-part-4-non-parametric-tests-4db7b4b6a974)
- [Power Analysis Made Easy](https://towardsdatascience.com/power-analysis-made-easy-dfee1eb813a)
- [Statistical Tests: When to Use Which](https://towardsdatascience.com/statistical-tests-when-to-use-which-704557554740)
- [The Art of A/B Testing](https://towardsdatascience.com/the-art-of-a-b-testing-5a10c9bb70a4)
- [Mann-Whitney U Test](https://towardsdatascience.com/determine-if-two-distributions-are-significantly-different-using-the-mann-whitney-u-test-1f79aa249ffb)
- [Hypothesis Test to Online Experiments](https://towardsdatascience.com/python-code-from-hypothesis-test-to-online-experiments-with-buiness-cases-e0597c6d1ec)
- [Implementing A/B Tests in Python](https://medium.com/@robbiegeoghegan/implementing-a-b-tests-in-python-514e9eb5b3a1)
- [Chi-Squared A/B Test Calculator](https://www.evanmiller.org/ab-testing/chi-squared.html)
- [Udacity A/B Testing Course Notes](https://github.com/baumanab/udacity_ABTesting#summary)
