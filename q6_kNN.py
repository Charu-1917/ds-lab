import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score)

df = pd.DataFrame({
    "Glucose":       [148,85,183,89,137,116,78,115,197,125,110,168,139,189,166,100,118,107,103,115,126,99,196,119],
    "BloodPressure": [72,66,64,66,40,74,50,0,70,96,92,74,80,60,72,0,84,74,30,70,88,84,90,80],
    "BMI":           [33.6,26.6,23.3,28.1,43.1,25.6,31.0,35.3,30.5,0,37.6,38.0,27.1,30.1,25.8,30.0,45.8,29.6,43.3,34.6,39.3,35.4,39.8,29.0],
    "Age":           [50,31,32,21,33,30,26,29,53,54,30,34,57,59,51,32,31,31,33,32,27,50,41,29],
    "Outcome":       [1,0,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,0,0,1,0,0,1,0]})

print(df.head())
df.info()
print(df.describe())
print(df["Outcome"].value_counts())

cols = ["Glucose", "BloodPressure", "BMI"]
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

for k in [3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors=k).fit(Xtr, ytr)
    p = knn.predict(Xte)
    prob = knn.predict_proba(Xte)[:, 1]


    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    print(k, accuracy_score(yte, p), precision_score(yte, p), recall_score(yte, p),
          "Spec:", tn / (tn + fp), "F1:", f1_score(yte, p), "AUC:", roc_auc_score(yte, prob))


    fpr, tpr, _ = roc_curve(yte, prob)
    plt.plot(fpr, tpr, label=f"K={k}")

plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(); plt.show()
