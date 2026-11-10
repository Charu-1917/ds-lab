import numpy as np, pandas as pd
from scipy import stats

df = pd.DataFrame({
    "Price":  [250, 400, 150, 500, 300, 450, 200, 350, 600, 300],
    "Sales":  [30, 20, 45, 15, 28, 18, 40, 25, 10, 27],
    "Rating": [4.5, 4.0, 4.8, 3.5, 4.2, 3.8, 4.6, 4.1, 3.2, 4.3],
    "Category": ["A", "B", "A", "B", "A", "B", "A", "B", "B", "A"],
    "Rating_Level": ["High", "Low", "High", "Low", "High", "Low", "High", "High", "Low", "High"]})

x = df["Price"]

print(x.mean(), x.median(), x.mode()[0], x.var(), x.std(), x.max() - x.min())
print("CV%:", x.std() / x.mean() * 100)

s = x.sample(5, random_state=1)
ci = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
print(ci)

print(x.skew(), x.kurt())

t, p = stats.ttest_1samp(x, 300)
print("Reject H0" if p < 0.05 else "Fail to reject H0")

print(df[["Price", "Sales"]].cov())
print(df[["Price", "Sales"]].corr())

z = np.abs(stats.zscore(x))
print(df[z > 3])

ct = pd.crosstab(df["Category"], df["Rating_Level"])
chi2, p, dof, exp = stats.chi2_contingency(ct)
print("Related" if p < 0.05 else "Independent")

r = stats.linregress(df["Price"], df["Sales"])
print(r.slope, r.intercept, r.rvalue**2, r.pvalue)