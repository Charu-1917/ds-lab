import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score)

X0, y0 = make_classification(n_samples=500, n_features=4, n_informative=3,
                             n_redundant=1, random_state=42)
df = pd.DataFrame(X0, columns=["Variance", "Skewness", "Curtosis", "Entropy"])
df["Class"] = y0

print(df.head()); df.info(); print(df.describe())
print(df.isnull().sum())
df = df.drop_duplicates()

def entropy(y):
    p = y.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

parent = entropy(df["Class"])
left = df[df["Variance"] > 0]
right = df[df["Variance"] <= 0]
child = len(left)/len(df)*entropy(left["Class"]) + len(right)/len(df)*entropy(right["Class"])
print("Entropy:", parent, "Info Gain:", parent - child)


X = df.drop("Class", axis=1)
y = df["Class"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)

for d in [None, 3, 5, 7]:
    dt = DecisionTreeClassifier(criterion="entropy", max_depth=d, random_state=42).fit(Xtr, ytr)
    p = dt.predict(Xte)
    prob = dt.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    print("depth", d, "Acc:", accuracy_score(yte, p), "Prec:", precision_score(yte, p),
          "Rec:", recall_score(yte, p), "Spec:", tn/(tn+fp),
          "F1:", f1_score(yte, p), "AUC:", roc_auc_score(yte, prob))
    fpr, tpr, _ = roc_curve(yte, prob)
    plt.plot(fpr, tpr, label=f"depth={d}")

    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.legend(); plt.show()
