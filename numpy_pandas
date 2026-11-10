import numpy as np, pandas as pd
students = {}
def add(i, name, marks): students[i] = {"name": name, "marks": marks}

def update(i, marks):
    if i in students: students[i]["marks"] = marks

def search(i): return students.get(i, "Not found")

def display():
    for i, s in students.items(): print(i, s["name"], s["marks"])

add(1, "Anu", (80, 75, 90)); add(2, "Ravi", (60, 70, 65)); add(3, "Sam", (90, 85, 95))
update(2, (65, 72, 68))
print(search(1))
display()

subjects = {"Maths", "Physics", "Chem"}
names = [s["name"] for s in students.values()]
marks = np.array([s["marks"] for s in students.values()])

print(marks[0], marks[1, 2], marks[:, 0], marks[:2, 1:])
print(np.sort(marks, axis=1))
print(marks.reshape(1, 9))

df = pd.DataFrame({"ID": list(students), "Name": names,
                   "M1": marks[:, 0], "M2": marks[:, 1], "M3": marks[:, 2]})

df["Total"] = df[["M1", "M2", "M3"]].sum(axis=1)
print(df.head(), df.describe())

df.to_csv("students.csv", index=False)