import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv(r"D:\Study\SUT-CS\Books\آمار\project\phase-1\cleaned data\NHANES_FINAL.csv")


# 2. Education labels
education_labels = {
    1: "Less than 9th grade",
    2: "9th-11th grade",
    3: "High school / GED",
    4: "Some college / Associate",
    5: "College graduate or above",
    7: "Refused",
    9: "Don't know"
}


# 3. Calculate sleep duration
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


df["Weekday_Sleep"] = sleep_duration_vectorized(
    df["SLQ300"], df["SLQ310"]
)

df["Weekend_Sleep"] = sleep_duration_vectorized(
    df["SLQ320"], df["SLQ330"]
)


# 4. Convert columns to numeric
df["DMDEDUC2"] = pd.to_numeric(
    df["DMDEDUC2"], errors="coerce"
)

df["WTINT2YR"] = pd.to_numeric(
    df["WTINT2YR"], errors="coerce"
)


# 5. Keep valid observations
df = df[
    df["DMDEDUC2"].isin([1, 2, 3, 4, 5]) &
    (df["WTINT2YR"] > 0)
].copy()


# 6. Weighted mean function
def weighted_mean(data, weight):

    valid = data.notna() & weight.notna() & (weight > 0)

    if valid.sum() == 0:
        return np.nan

    return np.average(
        data[valid],
        weights=weight[valid]
    )


# 7. Calculate average sleep for each education group
results = []

for education in [1, 2, 3, 4, 5]:

    group = df[df["DMDEDUC2"] == education]

    results.append({
        "Education": education_labels[education],

        "Weekday Sleep": weighted_mean(
            group["Weekday_Sleep"],
            group["WTINT2YR"]
        ),

        "Weekend Sleep": weighted_mean(
            group["Weekend_Sleep"],
            group["WTINT2YR"]
        ),

        "Sample Size": len(group)
    })


result_df = pd.DataFrame(results)


# 8. Draw grouped bar chart
x = np.arange(len(result_df))
width = 0.35

plt.figure(figsize=(13, 7))

bars1 = plt.bar(
    x - width/2,
    result_df["Weekday Sleep"],
    width,
    label="Weekday Sleep",
    edgecolor="black"
)

bars2 = plt.bar(
    x + width/2,
    result_df["Weekend Sleep"],
    width,
    label="Weekend Sleep",
    edgecolor="black"
)


# 9. Add values above bars
for bars in [bars1, bars2]:

    for bar in bars:

        height = bar.get_height()

        if np.isfinite(height):
            plt.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9
            )


plt.xticks(
    x,
    result_df["Education"],
    rotation=20,
    ha="right"
)

plt.ylabel("Weighted Average Sleep Duration (Hours)")
plt.xlabel("Education Level")

plt.title("Average Sleep Duration by Education Level")

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()


# 10. Print results
print("\nSleep Duration by Education Level:\n")

print(result_df.round(2).to_string(index=False))