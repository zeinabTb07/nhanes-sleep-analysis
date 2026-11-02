import pandas as pd

# 1. Read final dataset
df = pd.read_csv(
    r"D:\Study\SUT-CS\Books\آمار\project\phase-1\cleaned data\NHANES_FINAL.csv"
)

# تابع نوشتن در فایل MD به جای print
md_file = open("sleep_analysis.md", "w", encoding="utf-8")
def out(*args):
    text = " ".join(str(a) for a in args)
    print(text)
    md_file.write(text + "\n")

# 2. Calculate sleep duration in hours
def sleep_duration(bedtime, waketime):
    bed = int(bedtime[:2]) * 60 + int(bedtime[3:5])
    wake = int(waketime[:2]) * 60 + int(waketime[3:5])
    return ((wake - bed) % 1440) / 60


# Weekday sleep duration
df["Weekday_Sleep"] = df.apply(
    lambda row: sleep_duration(row["SLQ300"], row["SLQ310"]),
    axis=1
)

# Weekend sleep duration
df["Weekend_Sleep"] = df.apply(
    lambda row: sleep_duration(row["SLQ320"], row["SLQ330"]),
    axis=1
)


# 3. General descriptive statistics
out("\n========== GENERAL STATISTICS ==========")

for col in ["Weekday_Sleep", "Weekend_Sleep"]:
    out(f"\n{col}")
    out("Mean:", round(df[col].mean(), 2))
    out("Median:", round(df[col].median(), 2))
    out("Minimum:", round(df[col].min(), 2))
    out("Maximum:", round(df[col].max(), 2))
    out("Standard Deviation:", round(df[col].std(), 2))


# 4. Average sleep by gender
out("\n========== SLEEP BY GENDER ==========")
out(df.groupby("RIAGENDR")[["Weekday_Sleep", "Weekend_Sleep"]].mean().round(2))


# 5. Average sleep by age groups
out("\n========== SLEEP BY AGE GROUP ==========")

bins = [20, 23, 26, 29, 32, 36]
labels = ["20-22", "23-25", "26-28", "29-31", "32-35"]

df["Age_Group"] = pd.cut(df["RIDAGEYR"], bins=bins, labels=labels, right=False)

age_sleep = df.groupby("Age_Group", observed=False).agg(
    Participants=("SEQN", "count"),
    Weekday_Mean=("Weekday_Sleep", "mean"),
    Weekend_Mean=("Weekend_Sleep", "mean")
).round(2)

out(age_sleep)


# 6. Age frequency
out("\n========== AGE FREQUENCY ==========")
out(df["RIDAGEYR"].value_counts().sort_index())


# 8. Education statistics
education_labels = {
    1: "Less than 9th grade",
    2: "9th-11th grade",
    3: "High school / GED",
    4: "Some college / Associate",
    5: "College graduate or above",
    7: "Refused",
    9: "Don't know"
}

df["Education"] = pd.to_numeric(df["DMDEDUC2"], errors="coerce").map(education_labels)


# 9. Sleep by education
out("\n========== SLEEP BY EDUCATION ==========")

education_sleep = df.groupby("Education").agg(
    Participants=("SEQN", "count"),
    Weekday_Mean=("Weekday_Sleep", "mean"),
    Weekend_Mean=("Weekend_Sleep", "mean")
).round(2)

out(education_sleep)


# 10. Sleep by income groups
out("\n========== SLEEP BY INCOME ==========")

df["Income_Group"] = pd.cut(
    df["INDFMPIR"],
    bins=[-0.001, 1, 3, 5.001],
    labels=["Low", "Middle", "High"]
)

income_sleep = df.groupby("Income_Group", observed=False).agg(
    Participants=("SEQN", "count"),
    Weekday_Mean=("Weekday_Sleep", "mean"),
    Weekend_Mean=("Weekend_Sleep", "mean")
).round(2)

out(income_sleep)


# 11. Sleep duration categories
def sleep_category(hours):
    if hours < 6:
        return "Less than 6 hours"
    elif hours <= 9:
        return "6 to 9 hours"
    else:
        return "More than 9 hours"


df["Weekday_Category"] = df["Weekday_Sleep"].apply(sleep_category)
df["Weekend_Category"] = df["Weekend_Sleep"].apply(sleep_category)

out("\n========== SLEEP FREQUENCY ==========")
out("\nWeekday:")
out(df["Weekday_Category"].value_counts())
out("\nWeekend:")
out(df["Weekend_Category"].value_counts())


# 12. Gender frequency
out("\n========== GENDER FREQUENCY ==========")
out(df["RIAGENDR"].value_counts())


# 13. Total participants
out("\n========== SAMPLE SIZE ==========")
out("Total participants:", len(df))


# 14. Save analysis tables
age_sleep.to_csv("sleep_by_age.csv", encoding="utf-8-sig")
df.to_csv("NHANES_ANALYZED.csv", index=False, encoding="utf-8-sig")

md_file.close()
print("\nAnalysis completed successfully!")