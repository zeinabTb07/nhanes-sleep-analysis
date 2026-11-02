
import pandas as pd
import numpy as np

# 1. Read merged CSV
df = pd.read_csv(
    r"D:\Study\SUT-CS\Books\آمار\project\phase-1\cleaned data\NHANES_MARGED.csv"
)

before = len(df)

# 2. Convert age to numeric
df["RIDAGEYR"] = pd.to_numeric(
    df["RIDAGEYR"], errors="coerce"
)

# 3. Standardize sleep time format
sleep_cols = ["SLQ300", "SLQ310", "SLQ320", "SLQ330"]

for col in sleep_cols:
    df[col] = (
        df[col]
        .astype("string")
        .str.replace(r"^b['\"]|['\"]$", "", regex=True)
        .str.strip()
    )

# 4. Replace missing and special sleep responses
df[sleep_cols] = df[sleep_cols].replace(
    ["77777", "99999", ""], np.nan
)

# 5. Keep only participants aged 20 to 35
df = df[df["RIDAGEYR"].between(20, 35)].copy()

after_age = len(df)

# 6. Remove participants without complete sleep data
df = df.dropna(subset=sleep_cols).copy()

after_sleep = len(df)

# 7. Replace missing income with mean
df["INDFMPIR"] = pd.to_numeric(
    df["INDFMPIR"], errors="coerce"
)

df["INDFMPIR"] = df["INDFMPIR"].fillna(
    df["INDFMPIR"].mean()
)

# 8. Standardize gender labels
df["RIAGENDR"] = pd.to_numeric(
    df["RIAGENDR"], errors="coerce"
).map({
    1: "Male",
    2: "Female"
})

# 9. Save final dataset
df.to_csv(
    r"D:\Study\SUT-CS\Books\آمار\project\phase-1\cleaned data\NHANES_FINAL.csv",
    index=False,
    encoding="utf-8-sig"
)

# 10. Report
print("\n========== CLEANING REPORT ==========")
print("Initial participants:", before)
print("After age filtering:", after_age)
print("After sleep cleaning:", after_sleep)
print("Removed by age:", before - after_age)
print("Removed by missing sleep:", after_age - after_sleep)
print("Final participants:", len(df))

print("\nAge range in final dataset:")
print(df["RIDAGEYR"].min(), "-", df["RIDAGEYR"].max())

print("\nMissing sleep values:")
print(df[sleep_cols].isna().sum())

print("\nSaved: NHANES_FINAL.csv")
