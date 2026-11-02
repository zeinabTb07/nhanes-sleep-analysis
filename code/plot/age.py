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


# 4. Convert age and weight to numeric
df["RIDAGEYR"] = pd.to_numeric(
    df["RIDAGEYR"], errors="coerce"
)

df["WTINT2YR"] = pd.to_numeric(
    df["WTINT2YR"], errors="coerce"
)


# 5. Keep valid observations
df = df[
    df["RIDAGEYR"].notna() &
    df["WTINT2YR"].notna() &
    (df["WTINT2YR"] > 0)
].copy()


# 6. Weighted mean function
def weighted_mean(data, weights):

    valid = data.notna() & weights.notna() & (weights > 0)

    if valid.sum() == 0:
        return np.nan

    return np.average(
        data[valid],
        weights=weights[valid]
    )


# 7. Calculate weighted average sleep by age
results = []

for age in sorted(df["RIDAGEYR"].unique()):

    group = df[df["RIDAGEYR"] == age]

    results.append({
        "Age": age,

        "Weekday_Sleep": weighted_mean(
            group["Weekday_Sleep"],
            group["WTINT2YR"]
        ),

        "Weekend_Sleep": weighted_mean(
            group["Weekend_Sleep"],
            group["WTINT2YR"]
        )
    })


result_df = pd.DataFrame(results)


# 8. Draw line chart
plt.figure(figsize=(12, 6))

plt.plot(
    result_df["Age"],
    result_df["Weekday_Sleep"],
    marker="o",
    markersize=5,
    linewidth=2.5,
    label="Weekday Sleep"
)

plt.plot(
    result_df["Age"],
    result_df["Weekend_Sleep"],
    marker="o",
    markersize=5,
    linewidth=2.5,
    label="Weekend Sleep"
)

plt.xlim(20, 35)
plt.xticks(range(20, 36, 1))

plt.ylim(0, 12)
plt.yticks(range(0, 13, 1))

plt.xlabel("Age (Years)", fontsize=11)
plt.ylabel("Average Sleep Duration (Hours)", fontsize=11)

plt.title(
    "Average Sleep Duration by Age (20–35)",
    fontsize=14,
    pad=15
)

plt.grid(axis="both", linestyle="--", alpha=0.25)
plt.legend(frameon=False, fontsize=10)

plt.tight_layout()
plt.show()

# 9. Print results
print("\nWeighted Average Sleep by Age:\n")
print(result_df.round(2).to_string(index=False))