import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.DataFrame({
    "Area":     [800, 1000, 1200, 1500, 1800, 2000, 900, 1100, 1600, 2200, 1300, 1700],
    "Bedrooms": [2, 2, 3, 3, 4, 4, 2, np.nan, 3, 5, 3, 4],
    "Location": ["Urban", "Rural", "Urban", "Suburb", "Urban", "Suburb",
                 "Rural", "Suburb", "Rural", "Urban", "Urban", "Suburb"],
    "Price":    [40, 38, 62, 70, 95, 90, 30, 55, 60, 120, 65, 85]})

print(df.head())
df.info()
print(df.describe())
print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))

df = pd.get_dummies(df, columns=["Location"], drop_first=True)

X = df.drop("Price", axis=1)
y = df["Price"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

simple = LinearRegression().fit(Xtr[["Area"]], ytr)
print("Simple R2:", r2_score(yte, simple.predict(Xte[["Area"]])))

sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

model = LinearRegression().fit(Xtr, ytr)
pred = model.predict(Xte)
print(pred)

mae = mean_absolute_error(yte, pred)
mse = mean_squared_error(yte, pred)
print(mae, mse, np.sqrt(mse), r2_score(yte, pred))

plt.scatter(yte, pred)
plt.plot([yte.min(), yte.max()], [yte.min(), yte.max()], "r")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.show()