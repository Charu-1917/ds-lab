import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

df = pd.DataFrame({
    "Category": ["Electronics", "Clothing", "Grocery", "Furniture", "Toys", "Electronics",
                 "Clothing", "Grocery", "Furniture", "Toys", "Electronics", "Grocery"],
    "Month":    ["Jan", "Jan", "Jan", "Feb", "Feb", "Feb", "Mar", "Mar", "Mar", "Apr", "Apr", "Apr"],
    "Sales":    [500, 300, 200, 450, 150, 550, 320, 220, 400, 180, 600, 900],
    "Profit":   [100, 60, 30, 90, 20, 120, 65, 35, 80, 25, 140, 50],
    "Revenue":  [520, 310, 210, 460, 160, 570, 330, 230, 410, 190, 620, 910]})

cat = df.groupby("Category")["Sales"].sum()

cat.plot(kind="bar")
plt.title("Sales by Category"); plt.show()


df.groupby("Category")[["Sales", "Profit"]].sum().plot(kind="bar")
plt.show()

cat.plot(kind="barh")
plt.show()

plt.pie(cat, labels=cat.index, autopct="%1.1f%%")
plt.show()

df.groupby("Month", sort=False)["Sales"].sum().plot(marker="o")
plt.show()

sns.scatterplot(x="Sales", y="Profit", data=df)
plt.show()


sns.boxplot(y=df["Sales"])
plt.show()

plt.hist(df["Sales"], bins=5)
plt.show()

sns.pairplot(df[["Sales", "Profit", "Revenue"]])
plt.show()