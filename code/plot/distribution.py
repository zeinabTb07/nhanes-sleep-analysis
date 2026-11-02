import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv(r"D:\Study\SUT-CS\Books\آمار\project\phase-1\cleaned data\NHANES_FINAL.csv")


# 2. Calculate sleep duration
def sleep_duration_vectorized(bedtime, waketime):

    bed = pd.to_timedelta(
        bedtime.astype(str) + ":00",
        errors="coerce"
    )

    wake = pd.to_timedelta(
        waketime.astype(str) + ":00",
        errors="coerce"
    )

    duration = (wake - bed) % pd.Timedelta(days=1)

    return duration.dt.total_seconds() / 3600


# 3. Calculate weekday and weekend sleep
df["Weekday_Sleep"] = sleep_duration_vectorized(
    df["SLQ300"], df["SLQ310"]
)

df["Weekend_Sleep"] = sleep_duration_vectorized(
    df["SLQ320"], df["SLQ330"]
)


# 4. Weighted quantile function
def weighted_quantile(values, weights, quantiles):

    values = np.asarray(values)
    weights = np.asarray(weights)

    valid = (
        np.isfinite(values) &
        np.isfinite(weights) &
        (weights > 0)
    )

    values = values[valid]
    weights = weights[valid]

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative_weights = np.cumsum(weights)
    cumulative_weights = cumulative_weights / cumulative_weights[-1]

    return np.interp(
        quantiles,
        cumulative_weights,
        values
    )


# 5. Prepare weighted boxplot statistics
def weighted_boxplot_stats(data, weight):

    valid = data.notna() & weight.notna() & (weight > 0)

    values = data[valid].to_numpy()
    weights = weight[valid].to_numpy()

    q1, median, q3 = weighted_quantile(
        values,
        weights,
        [0.25, 0.50, 0.75]
    )

    iqr = q3 - q1

    lower_whisker = values[values >= q1 - 1.5 * iqr].min()
    upper_whisker = values[values <= q3 + 1.5 * iqr].max()

    mean = np.average(values, weights=weights)

    return {
        "label": "",
        "med": median,
        "q1": q1,
        "q3": q3,
        "whislo": lower_whisker,
        "whishi": upper_whisker,
        "mean": mean,
        "fliers": values[
            (values < lower_whisker) |
            (values > upper_whisker)
        ]
    }


# 6. Calculate statistics
weekday_stats = weighted_boxplot_stats(
    df["Weekday_Sleep"],
    df["WTINT2YR"]
)

weekend_stats = weighted_boxplot_stats(
    df["Weekend_Sleep"],
    df["WTINT2YR"]
)

# 7. Draw weighted boxplots without bxp

plt.figure(figsize=(9, 6))

stats = [weekday_stats, weekend_stats]
positions = [1, 2]

for pos, s in zip(positions, stats):

    # Box
    plt.fill_between(
        [pos - 0.25, pos + 0.25],
        s["q1"],
        s["q3"],
        alpha=0.4
    )

    # Median
    plt.plot(
        [pos - 0.25, pos + 0.25],
        [s["med"], s["med"]],
        color="black",
        linewidth=2
    )

    # Whiskers
    plt.plot(
        [pos, pos],
        [s["whislo"], s["q1"]],
        color="black"
    )

    plt.plot(
        [pos, pos],
        [s["q3"], s["whishi"]],
        color="black"
    )

    # Whisker caps
    plt.plot(
        [pos - 0.12, pos + 0.12],
        [s["whislo"], s["whislo"]],
        color="black"
    )

    plt.plot(
        [pos - 0.12, pos + 0.12],
        [s["whishi"], s["whishi"]],
        color="black"
    )

    # Mean
    plt.scatter(
        pos,
        s["mean"],
        marker="x",
        color="red",
        zorder=3
    )

    # Outliers
    plt.scatter(
        [pos] * len(s["fliers"]),
        s["fliers"],
        marker="o",
        s=12,
        alpha=0.4
    )


plt.xticks(
    positions,
    ["Weekday Sleep", "Weekend Sleep"]
)

plt.ylabel("Sleep Duration (Hours)")
plt.title("Weighted Sleep Duration: Weekdays vs Weekends")

plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()

# 8. Weighted descriptive statistics
def weighted_summary(data, weight):

    valid = data.notna() & weight.notna() & (weight > 0)

    values = data[valid].to_numpy()
    weights = weight[valid].to_numpy()

    q1, median, q3 = weighted_quantile(
        values,
        weights,
        [0.25, 0.50, 0.75]
    )

    mean = np.average(values, weights=weights)

    std = np.sqrt(
        np.average((values - mean) ** 2, weights=weights)
    )

    return {
        "Mean": mean,
        "Median": median,
        "Std": std,
        "Q1": q1,
        "Q3": q3,
        "Min": values.min(),
        "Max": values.max()
    }


summary = pd.DataFrame({
    "Weekday": weighted_summary(
        df["Weekday_Sleep"], df["WTINT2YR"]
    ),
    "Weekend": weighted_summary(
        df["Weekend_Sleep"], df["WTINT2YR"]
    )
})

print("\nWeighted Sleep Statistics:\n")
print(summary.round(2))