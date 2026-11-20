
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import binom

WEIGHT_COL = "WTINT2YR"


# ==================================================
# 1. Time utilities
# ==================================================

def time_to_minutes(value):
    if pd.isna(value):
        return np.nan

    try:
        h, m = map(int, str(value).strip().split(":"))

        if not (0 <= h < 24 and 0 <= m < 60):
            return np.nan

        return h * 60 + m

    except (ValueError, AttributeError):
        return np.nan


def add_sleep_features(df):
    data = df.copy()

    time_cols = ["SLQ300", "SLQ310", "SLQ320", "SLQ330"]

    for col in time_cols:
        data[col] = data[col].apply(time_to_minutes)

    # Sleep duration
    data["Weekday_Sleep"] = (
        (data["SLQ310"] - data["SLQ300"]) % 1440
    ) / 60

    data["Weekend_Sleep"] = (
        (data["SLQ330"] - data["SLQ320"]) % 1440
    ) / 60

    # Invalid zero-duration observations
    data.loc[data["Weekday_Sleep"] == 0, "Weekday_Sleep"] = np.nan
    data.loc[data["Weekend_Sleep"] == 0, "Weekend_Sleep"] = np.nan

    # Weekend sleep midpoint
    data["Sleep_Midpoint"] = (
        data["SLQ320"] + data["Weekend_Sleep"] * 30
    ) % 1440 / 60

    return data


# ==================================================
# 2. Weighted statistical utilities
# ==================================================

def weighted_mean(values, weights):
    valid = (
        values.notna()
        & weights.notna()
        & np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return np.average(
        values[valid],
        weights=weights[valid]
    )


def weighted_probability(event, weights, valid=None):
    if valid is None:
        valid = event.notna()

    valid = (
        valid
        & event.notna()
        & weights.notna()
        & np.isfinite(weights)
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return np.average(
        event[valid].astype(float),
        weights=weights[valid]
    )


def weighted_conditional_probability(A, B, weights):
    valid = (
        A.notna()
        & B.notna()
        & weights.notna()
        & np.isfinite(weights)
        & (weights > 0)
    )

    denominator = weights[valid & A].sum()

    if denominator == 0:
        return np.nan

    numerator = weights[valid & A & B].sum()

    return numerator / denominator


# ==================================================
# 3. Weekend Sleep Compensation
# ==================================================

def analyze_sleep_compensation(df, extra_hours=1):

    x = df["Weekday_Sleep"]
    y = df["Weekend_Sleep"]
    w = df[WEIGHT_COL]

    valid = (
        x.notna()
        & y.notna()
        & w.notna()
        & np.isfinite(w)
        & (w > 0)
    )

    # Weighted means on the same complete-case sample
    mean_weekday = weighted_mean(x[valid], w[valid])
    mean_weekend = weighted_mean(y[valid], w[valid])

    # Weighted thresholds
    threshold_a = mean_weekday - extra_hours
    threshold_b = mean_weekend + extra_hours

    A = x < threshold_a
    B = y > threshold_b

    # Weighted probabilities
    p_a = weighted_probability(A, w, valid)
    p_b = weighted_probability(B, w, valid)
    p_ab = weighted_probability(A & B, w, valid)

    p_b_given_a = weighted_conditional_probability(A, B, w)

    # Sample counts and weighted totals
    n = int(valid.sum())
    weight_total = w[valid].sum()
    weight_a = w[valid & A].sum()
    weight_b = w[valid & B].sum()
    weight_ab = w[valid & A & B].sum()

    summary = pd.DataFrame({
        "Metric": [
            "Valid sample size",
            "Total survey weight",
            "Weighted weekday mean",
            "Weighted weekend mean",
            "Event A threshold",
            "Event B threshold",
            "N(A)",
            "N(B)",
            "N(A intersection B)",
            "Weighted P(A)",
            "Weighted P(B)",
            "Weighted P(A intersection B)",
            "Weighted P(B | A)"
        ],
        "Value": [
            n,
            weight_total,
            mean_weekday,
            mean_weekend,
            threshold_a,
            threshold_b,
            int((valid & A).sum()),
            int((valid & B).sum()),
            int((valid & A & B).sum()),
            p_a,
            p_b,
            p_ab,
            p_b_given_a
        ]
    })

    events = pd.DataFrame({
        "A": A & valid,
        "B": B & valid,
        "Compensation": A & B & valid
    }, index=df.index)

    return {
        "summary": summary,
        "events": events,
        "thresholds": {
            "A": threshold_a,
            "B": threshold_b
        }
    }


# ==================================================
# 4. Night Owls
# ==================================================

def analyze_night_owls(df, midpoint_threshold=5):

    z = df["Sleep_Midpoint"]
    w = df[WEIGHT_COL]

    valid = (
        z.notna()
        & w.notna()
        & np.isfinite(w)
        & (w > 0)
    )

    night_owl = z >= midpoint_threshold

    probability = weighted_probability(
        night_owl,
        w,
        valid
    )

    summary = pd.DataFrame({
        "Metric": [
            "Valid sample size",
            "Night Owl sample count",
            "Night Owl threshold",
            "Weighted Night Owl probability",
            "Weighted Night Owl percentage"
        ],
        "Value": [
            int(valid.sum()),
            int((valid & night_owl).sum()),
            midpoint_threshold,
            probability,
            probability * 100 if pd.notna(probability) else np.nan
        ]
    })

    events = pd.DataFrame({
        "Night_Owl": night_owl & valid
    }, index=df.index)

    return {
        "summary": summary,
        "events": events
    }


# ==================================================
# 5. Weighted group analysis
# ==================================================

def compensation_by_group(df, group_column, extra_hours=1):

    overall = analyze_sleep_compensation(df, extra_hours)
    events = overall["events"]

    temp = df[[group_column, WEIGHT_COL]].copy()
    temp["A"] = events["A"]
    temp["B"] = events["B"]

    temp = temp.dropna(subset=[group_column, WEIGHT_COL])
    temp = temp[temp[WEIGHT_COL] > 0]

    rows = []

    for group, part in temp.groupby(group_column):

        p_a = weighted_probability(
            part["A"],
            part[WEIGHT_COL]
        )

        p_b_given_a = weighted_conditional_probability(
            part["A"],
            part["B"],
            part[WEIGHT_COL]
        )

        rows.append({
            "Group": group,
            "N": len(part),
            "Weighted P(A)": p_a,
            "Weighted P(B | A)": p_b_given_a
        })

    return pd.DataFrame(rows)


def night_owls_by_group(df, group_column, midpoint_threshold=3):

    temp = df[
        [group_column, "Sleep_Midpoint", WEIGHT_COL]
    ].dropna()

    temp = temp[temp[WEIGHT_COL] > 0]

    temp["Night_Owl"] = (
        temp["Sleep_Midpoint"] >= midpoint_threshold
    )

    rows = []

    for group, part in temp.groupby(group_column):

        probability = weighted_probability(
            part["Night_Owl"],
            part[WEIGHT_COL]
        )

        rows.append({
            "Group": group,
            "N": len(part),
            "Weighted Probability": probability,
            "Weighted Percentage": probability * 100
        })

    return pd.DataFrame(rows)


# ==================================================
# 6. Weighted descriptive statistics
# ==================================================

def weighted_sleep_statistics(df):

    rows = []

    for col in ["Weekday_Sleep", "Weekend_Sleep"]:

        valid = (
            df[col].notna()
            & df[WEIGHT_COL].notna()
            & (df[WEIGHT_COL] > 0)
        )

        values = df.loc[valid, col]
        weights = df.loc[valid, WEIGHT_COL]

        mean = weighted_mean(values, weights)

        variance = np.average(
            (values - mean) ** 2,
            weights=weights
        ) if len(values) else np.nan

        rows.append({
            "Variable": col,
            "Weighted Mean": mean,
            "Weighted Variance": variance,
            "Weighted Std": np.sqrt(variance),
            "N": len(values)
        })

    return pd.DataFrame(rows)


# ==================================================
# 7. Export
# ==================================================

def save_results(results, output_dir, prefix):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, value in results.items():

        if isinstance(value, pd.DataFrame):

            value.to_csv(
                output_dir / f"{prefix}_{name}.csv",
                index=False
            )



# ============================================================
# PHASE 2 - ADDITIONAL STATISTICAL ANALYSES
# ============================================================

TIME_COLUMNS = ["SLQ300", "SLQ310", "SLQ320", "SLQ330"]


def _valid_weights(data):
    """Keep observations with valid positive survey weights."""
    return (
        data["WTINT2YR"].notna()
        & np.isfinite(data["WTINT2YR"])
        & (data["WTINT2YR"] > 0)
    )


def _weighted_mean(values, weights):
    """Weighted mean."""
    return np.sum(values * weights) / np.sum(weights)


def _weighted_variance(values, weights):
    """Descriptive weighted population variance."""
    mean = _weighted_mean(values, weights)
    return np.sum(weights * (values - mean) ** 2) / np.sum(weights)


def _weighted_probability(event, weights):
    """Weighted probability of a Boolean event."""
    return weights[event].sum() / weights.sum()


# ------------------------------------------------------------
# 5. JOINT DISTRIBUTION: NIGHT OWLS AND EDUCATION
# ------------------------------------------------------------

def night_owl_education_distribution(df):
    """
    Weighted joint distribution of Night Owl status and education.

    Night Owl:
        Sleep_Midpoint >= 3

    Education:
        DMDEDUC2 categories
        Codes 7 and 9 are excluded as non-substantive responses.
    """

    data = df[
        ["Sleep_Midpoint", "DMDEDUC2", "WTINT2YR"]
    ].copy()

    data = data.dropna(
        subset=["Sleep_Midpoint", "DMDEDUC2", "WTINT2YR"]
    )

    data = data[_valid_weights(data)]

    # Keep valid education categories
    data = data[data["DMDEDUC2"].isin([1, 2, 3, 4, 5])]

    data["Night_Owl"] = data["Sleep_Midpoint"] >= 3

    data["Night_Owl"] = data["Night_Owl"].map({
        True: "Night Owl",
        False: "Not Night Owl"
    })

    # Weighted joint frequency
    joint_weights = pd.crosstab(
        data["Night_Owl"],
        data["DMDEDUC2"],
        values=data["WTINT2YR"],
        aggfunc="sum"
    ).fillna(0)

    # Convert weighted frequencies to joint probabilities
    joint_probabilities = joint_weights / joint_weights.values.sum()

    return {
        "weighted_counts": joint_weights,
        "joint_probabilities": joint_probabilities,
        "raw_counts": pd.crosstab(
            data["Night_Owl"], data["DMDEDUC2"]
        ),
        "sample_size": len(data)
    }


# ------------------------------------------------------------
# 5. BINOMIAL APPROXIMATION: NIGHT OWLS
# ------------------------------------------------------------

def analyze_binomial_night_owls(df, n=100, seed=42):
    """
    Approximate the number of Night Owls in a sample
    using a Binomial distribution.

    p is estimated using survey weights.
    The simulated sample assumes independent Bernoulli trials.
    """

    data = df[
        ["Sleep_Midpoint", "WTINT2YR"]
    ].dropna().copy()

    data = data[_valid_weights(data)]

    data["Night_Owl"] = data["Sleep_Midpoint"] >= 3

    p = _weighted_probability(
        data["Night_Owl"],
        data["WTINT2YR"]
    )

    rng = np.random.default_rng(seed)

    # One simulated sample of size n
    observed_count = rng.binomial(n, p)

    # Theoretical binomial probability distribution
    k = np.arange(n + 1)

    pmf = binom.pmf(k, n, p)

    distribution = pd.DataFrame({
        "Number_of_Night_Owls": k,
        "Probability": pmf
    })

    return {
        "p": p,
        "n": n,
        "expected_count": n * p,
        "variance": n * p * (1 - p),
        "simulated_count": observed_count,
        "distribution": distribution
    }


# ------------------------------------------------------------
# 6. NORMAL DISTRIBUTION: WEEKDAY SLEEP
# ------------------------------------------------------------

def analyze_weekday_sleep_distribution(df):
    """
    Weighted descriptive statistics and shape indicators
    for Weekday_Sleep.
    """

    data = df[
        ["Weekday_Sleep", "WTINT2YR"]
    ].dropna().copy()

    data = data[_valid_weights(data)]

    x = data["Weekday_Sleep"].to_numpy(dtype=float)
    w = data["WTINT2YR"].to_numpy(dtype=float)

    mean = _weighted_mean(x, w)
    variance = _weighted_variance(x, w)
    std = np.sqrt(variance)

    if std == 0:
        skewness = np.nan
        kurtosis = np.nan
    else:
        z = (x - mean) / std

        skewness = np.sum(w * z ** 3) / np.sum(w)
        kurtosis = np.sum(w * z ** 4) / np.sum(w) - 3

    return {
        "mean": mean,
        "variance": variance,
        "std": std,
        "skewness": skewness,
        "excess_kurtosis": kurtosis,
        "sample_size": len(data),
        "data": data
    }


# ------------------------------------------------------------
# 7. CONDITIONAL EXPECTED VALUE BY AGE GROUP
# ------------------------------------------------------------

def expected_weekday_sleep_by_age(df):
    """
    Weighted expected weekday sleep duration
    for age groups:
        20 <= age < 25
        25 <= age < 30
        30 <= age < 35
    """

    data = df[
        ["RIDAGEYR", "Weekday_Sleep", "WTINT2YR"]
    ].dropna().copy()

    data = data[_valid_weights(data)]

    bins = [20, 25, 30, 35]
    labels = ["20-24", "25-29", "30-34"]

    data["Age_Group"] = pd.cut(
        data["RIDAGEYR"],
        bins=bins,
        labels=labels,
        right=False
    )

    data = data.dropna(subset=["Age_Group"])

    results = []

    for group, subset in data.groupby("Age_Group", observed=True):

        x = subset["Weekday_Sleep"].to_numpy(dtype=float)
        w = subset["WTINT2YR"].to_numpy(dtype=float)

        results.append({
            "Age_Group": str(group),
            "Weighted_Mean_Sleep": _weighted_mean(x, w),
            "Weighted_Variance": _weighted_variance(x, w),
            "Raw_Count": len(subset),
            "Weighted_Count": w.sum()
        })

    return pd.DataFrame(results)


# ------------------------------------------------------------
# 8. COVARIANCE: WEEKDAY AND WEEKEND SLEEP
# ------------------------------------------------------------

def analyze_sleep_covariance(df):
    """
    Weighted covariance and descriptive correlation
    between weekday and weekend sleep.
    """

    data = df[
        ["Weekday_Sleep", "Weekend_Sleep", "WTINT2YR"]
    ].dropna().copy()

    data = data[_valid_weights(data)]

    x = data["Weekday_Sleep"].to_numpy(dtype=float)
    y = data["Weekend_Sleep"].to_numpy(dtype=float)
    w = data["WTINT2YR"].to_numpy(dtype=float)

    mean_x = _weighted_mean(x, w)
    mean_y = _weighted_mean(y, w)

    variance_x = _weighted_variance(x, w)
    variance_y = _weighted_variance(y, w)

    covariance = np.sum(
        w * (x - mean_x) * (y - mean_y)
    ) / np.sum(w)

    if variance_x > 0 and variance_y > 0:
        correlation = covariance / np.sqrt(
            variance_x * variance_y
        )
    else:
        correlation = np.nan

    return {
        "mean_weekday": mean_x,
        "mean_weekend": mean_y,
        "variance_weekday": variance_x,
        "variance_weekend": variance_y,
        "covariance": covariance,
        "correlation": correlation,
        "sample_size": len(data),
        "data": data
    }


# ------------------------------------------------------------
# 9. BAYES: NIGHT OWL AND SAVINGS
# ------------------------------------------------------------

def analyze_savings_bayes(df):
    """
    Joint distribution and conditional probabilities
    for savings and Night Owl status.

    INQ300:
        1 = No savings
        2 = Has savings

    Night Owl:
        Sleep_Midpoint >= 3
    """

    data = df[
        ["INQ300", "Sleep_Midpoint", "WTINT2YR"]
    ].dropna().copy()

    data = data[_valid_weights(data)]

    data = data[data["INQ300"].isin([1, 2])]

    data["Savings"] = data["INQ300"].map({
        1: "No Savings",
        2: "Has Savings"
    })

    data["Night_Owl"] = np.where(
        data["Sleep_Midpoint"] >= 3,
        "Night Owl",
        "Not Night Owl"
    )

    # Weighted joint counts
    joint_weights = pd.crosstab(
        data["Savings"],
        data["Night_Owl"],
        values=data["WTINT2YR"],
        aggfunc="sum"
    ).fillna(0)

    joint_probabilities = (
        joint_weights / joint_weights.values.sum()
    )

    # Marginal probabilities
    p_savings = joint_probabilities.loc[
        "Has Savings"
    ].sum()

    p_no_savings = joint_probabilities.loc[
        "No Savings"
    ].sum()

    p_owl = joint_probabilities[
        "Night Owl"
    ].sum()

    p_not_owl = joint_probabilities[
        "Not Night Owl"
    ].sum()

    # Conditional probabilities
    p_owl_given_savings = (
        joint_probabilities.loc[
            "Has Savings", "Night Owl"
        ] / p_savings
    )

    p_owl_given_no_savings = (
        joint_probabilities.loc[
            "No Savings", "Night Owl"
        ] / p_no_savings
    )

    p_savings_given_owl = (
        joint_probabilities.loc[
            "Has Savings", "Night Owl"
        ] / p_owl
    )

    p_savings_given_not_owl = (
        joint_probabilities.loc[
            "Has Savings", "Not Night Owl"
        ] / p_not_owl
    )

    return {
        "weighted_counts": joint_weights,
        "joint_probabilities": joint_probabilities,
        "P_Night_Owl_given_Savings": p_owl_given_savings,
        "P_Night_Owl_given_No_Savings": p_owl_given_no_savings,
        "P_Savings_given_Night_Owl": p_savings_given_owl,
        "P_Savings_given_Not_Night_Owl": p_savings_given_not_owl,
        "P_Savings": p_savings,
        "P_No_Savings": p_no_savings,
        "P_Night_Owl": p_owl,
        "P_Not_Night_Owl": p_not_owl,
        "raw_counts": pd.crosstab(
            data["Savings"], data["Night_Owl"]
        ),
        "sample_size": len(data)
    }


# ------------------------------------------------------------
# 10. INDEPENDENCE: GENDER AND UNROUNDED SLEEP TIMES
# ------------------------------------------------------------

def analyze_sleep_rounding_independence(df):
    """
    Check the relationship between gender and having
    at least one unrounded time among the four sleep times.

    Unrounded means the minute component is not 00 or 30.

    All four time fields must be valid.
    """

    data = df[
        ["RIAGENDR", "WTINT2YR"] + TIME_COLUMNS
    ].copy()

    def to_minutes(value):

        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        try:
            h, m = map(int, str(value).strip().split(":"))

            if 0 <= h < 24 and 0 <= m < 60:
                return h * 60 + m

        except (ValueError, AttributeError):
            pass

        return np.nan

    for col in TIME_COLUMNS:
        data[col] = data[col].apply(to_minutes)

    # Require all four valid time values
    data = data.dropna(
        subset=["RIAGENDR", "WTINT2YR"] + TIME_COLUMNS
    )

    data = data[_valid_weights(data)]

    # At least one time is not on a 00 or 30 minute mark
    rounded = data[TIME_COLUMNS] % 30 == 0

    data["Unrounded"] = (~rounded).any(axis=1)

    data["Gender"] = data["RIAGENDR"].astype(str)

    # Weighted contingency table
    joint_weights = pd.crosstab(
        data["Gender"],
        data["Unrounded"].map({
            True: "Unrounded",
            False: "Rounded"
        }),
        values=data["WTINT2YR"],
        aggfunc="sum"
    ).fillna(0)

    joint_probabilities = (
        joint_weights / joint_weights.values.sum()
    )

    # Marginal probabilities
    p_unrounded = joint_probabilities[
        "Unrounded"
    ].sum()

    p_gender = joint_probabilities.sum(axis=1)

    # Joint probabilities and independence differences
    independence = []

    for gender in joint_probabilities.index:

        p_joint = joint_probabilities.loc[
            gender, "Unrounded"
        ]

        p_product = p_gender[gender] * p_unrounded

        independence.append({
            "Gender": gender,
            "P_Gender": p_gender[gender],
            "P_Unrounded_Given_Gender": (
                p_joint / p_gender[gender]
            ),
            "P_Joint": p_joint,
            "P_Product": p_product,
            "Difference": p_joint - p_product
        })

    return {
        "weighted_counts": joint_weights,
        "joint_probabilities": joint_probabilities,
        "independence_comparison": pd.DataFrame(independence),
        "raw_counts": pd.crosstab(
            data["Gender"], data["Unrounded"]
        ),
        "sample_size": len(data),
        "data": data
    }

