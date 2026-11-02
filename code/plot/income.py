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


# 4. Convert income ratio to numeric
df["INDFMPIR"] = pd.to_numeric(
    df["INDFMPIR"], errors="coerce"
)


# 5. Remove missing values
data = df.dropna(
    subset=[
        "INDFMPIR",
        "Weekday_Sleep",
        "Weekend_Sleep"
    ]
).copy()

data = data[data["INDFMPIR"] >= 0]


# 6. Group income into intervals
data["Income_Group"] = pd.cut(
    data["INDFMPIR"],
    bins=np.arange(0, 6.5, 0.5)
)

grouped = data.groupby(
    "Income_Group",
    observed=True
)

trend = grouped.agg(
    Income=("INDFMPIR", "mean"),
    Weekday=("Weekday_Sleep", "mean"),
    Weekend=("Weekend_Sleep", "mean")
).dropna()


# 7. Draw scatter plot
plt.figure(figsize=(12, 7))

plt.scatter(
    data["INDFMPIR"],
    data["Weekday_Sleep"],
    alpha=0.3,
    s=20,
    label="Weekday Sleep"
)

plt.scatter(
    data["INDFMPIR"],
    data["Weekend_Sleep"],
    alpha=0.3,
    s=20,
    label="Weekend Sleep"
)


# 8. Draw average trend lines
plt.plot(
    trend["Income"],
    trend["Weekday"],
    marker="o",
    linewidth=2.5,
    label="Weekday Average"
)

plt.plot(
    trend["Income"],
    trend["Weekend"],
    marker="o",
    linewidth=2.5,
    label="Weekend Average"
)


# 9. Chart settings
plt.xlabel("Income-to-Poverty Ratio")
plt.ylabel("Sleep Duration (Hours)")

plt.title("Income-to-Poverty Ratio vs Sleep Duration")

plt.grid(alpha=0.25, linestyle="--")
plt.legend()

plt.tight_layout()
plt.show()