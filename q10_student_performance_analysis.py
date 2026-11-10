import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, confusion_matrix)

np.random.seed(42)
n = 150
df = pd.DataFrame({"Attendance": np.random.randint(50, 100, n),
                   "Internal": np.random.randint(20, 50, n),
                   "Assignment": np.random.randint(5, 20, n)})
df["Final"] = (0.3*df["Attendance"] + 0.8*df["Internal"] + 1.5*df["Assignment"]
               - 15 + np.random.normal(0, 5, n)).clip(0, 100).round()


print(df.isnull().sum(), df.duplicated().sum())
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))


print(df.shape); df.info(); print(df.describe())


df["Avg"] = np.mean(df[["Internal", "Assignment"]], axis=1)

sns.histplot(df["Final"]); plt.show()
sns.scatterplot(x="Attendance", y="Final", data=df); plt.show()
sns.boxplot(data=df[["Attendance", "Internal", "Assignment", "Final"]]); plt.show()

print(df.describe())
sns.heatmap(df.corr(numeric_only=True), annot=True); plt.show()

X = df[["Attendance", "Internal", "Assignment"]]
y = df["Final"]
c = pd.cut(df["Final"], bins=[0, 40, 70, 100], labels=["Fail", "Average", "Good"], include_lowest=True)

Xtr, Xte, ytr, yte, ctr, cte = train_test_split(X, y, c, test_size=0.25, random_state=42)
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

lr = LinearRegression().fit(Xtr, ytr)
p = lr.predict(Xte)
mse = mean_squared_error(yte, p)
print("MAE:", mean_absolute_error(yte, p), "MSE:", mse, "RMSE:", np.sqrt(mse), "R2:", r2_score(yte, p))

dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xtr, ctr)
cp = dt.predict(Xte)
print("Accuracy:", accuracy_score(cte, cp))
print(confusion_matrix(cte, cp))

plt.scatter(yte, p)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.show()
