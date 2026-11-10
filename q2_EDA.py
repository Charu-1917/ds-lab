import numpy as np, pandas as pd

df = pd.read_csv("students.csv")

df = pd.DataFrame({
    "Name": ["Anu", "Ravi", "Sam", "Meena", "Ravi", "Dev"],
    "Department": ["CS", "IT", "CS", "IT", "IT", "CS"],
    "Attendance": [90, 75, None, 85, 75, 60],
    "Internal": [45, 38, 48, 40, 38, 30],
    "Final": [78, 55, 90, 65, 55, 35]})

print(df.shape, df.columns, df.dtypes)
df.info()
print(df.describe())

print(df.isnull().sum())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df = df.dropna()

print(df.duplicated().sum())
df = df.drop_duplicates()

print(df[["Name", "Final"]])
print(df[df["Final"] > 60])
df = df.sort_values("Final", ascending=False)

df["Result"] = np.where(df["Final"] >= 40, "Pass", "Fail")
df["Total"] = df["Internal"] + df["Final"]

print(df.groupby("Department")[["Internal", "Final"]].mean())

df.to_csv("cleaned.csv", index=False)