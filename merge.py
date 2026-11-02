
import pandas as pd

# 1. Load datasets
demo = pd.read_sas(
    r"D:\Study\SUT-CS\Books\آمار\project\phase-1\raw data\DEMO_L.xpt"
)

inq = pd.read_sas(
    r"D:\Study\SUT-CS\Books\آمار\project\phase-1\raw data\INQ_L.xpt"
)

slq = pd.read_sas(
    r"D:\Study\SUT-CS\Books\آمار\project\phase-1\raw data\SLQ_L.xpt"
)


demo = demo[
    ["SEQN", "INDFMPIR", "WTINT2YR",
     "RIDAGEYR", "RIAGENDR", "DMDEDUC2"]
]

inq = inq[["SEQN", "INQ300"]]

slq = slq[
    ["SEQN", "SLQ300", "SLQ310", "SLQ320", "SLQ330"]
]

# 3. Merge datasets
final_data = demo.merge(
    inq, on="SEQN", how="left",
    validate="one_to_one"
)

final_data = final_data.merge(
    slq, on="SEQN", how="left",
    validate="one_to_one"
)

# 4. Sort
final_data = final_data.sort_values("SEQN").reset_index(drop=True)

# 5. Save
final_data.to_csv(r"D:\Study\SUT-CS\Books\آمار\project\phase-1\cleaned data\NHANES_FINAL.csv", index=False)
import os

# 6. Report
print("Merge completed!")
print("Shape:", final_data.shape)
print(final_data.head())
print("\nMissing values:")
print(final_data.isnull().sum())
