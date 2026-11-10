import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

X0, _ = make_blobs(n_samples=200, centers=5, n_features=3, random_state=42)
d = pd.DataFrame(X0, columns=["Age", "Annual Income", "Spending Score"])
d = (d - d.min()) / (d.max() - d.min())
df = pd.DataFrame({"Age": (18 + d["Age"] * 52).astype(int),
                   "Annual Income": (15 + d["Annual Income"] * 120).astype(int),
                   "Spending Score": (1 + d["Spending Score"] * 98).astype(int)})

print(df.head()); df.info(); print(df.describe())

X = StandardScaler().fit_transform(df[["Age", "Annual Income", "Spending Score"]])

wcss = []
for k in range(1, 11):
    wcss.append(KMeans(n_clusters=k, random_state=42, n_init=10).fit(X).inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("K"); plt.ylabel("WCSS"); plt.show()

km = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X)
print(km.cluster_centers_)


plt.scatter(df["Annual Income"], df["Spending Score"], c=df["Cluster"])
plt.xlabel("Annual Income"); plt.ylabel("Spending Score"); plt.show()

print("Silhouette:", silhouette_score(X, df["Cluster"]))
print("Davies-Bouldin:", davies_bouldin_score(X, df["Cluster"]))

print(df.groupby("Cluster")[["Age", "Annual Income", "Spending Score"]].mean())