
import pandas as pd
import numpy as np

# Generate reproducible sample data
np.random.seed(42)

n = 1000

data = {
    "Study_Hours": np.random.randint(1, 13, n),
    "Sleep_Hours": np.random.randint(3, 10, n),
    "Attendance": np.random.randint(40, 101, n),
    "Breaks": np.random.randint(0, 7, n),
    "Stress_Level": np.random.randint(1, 11, n),
    "Workload": np.random.randint(1, 11, n)
}

df = pd.DataFrame(data)

# Create illustrative academic strain risk labels
def calculate_risk(row):
    score = 0

    if row["Study_Hours"] > 8:
        score += 2

    if row["Sleep_Hours"] < 6:
        score += 2

    if row["Attendance"] < 65:
        score += 1

    if row["Breaks"] < 2:
        score += 1

    if row["Stress_Level"] >= 8:
        score += 2

    if row["Workload"] >= 8:
        score += 1

    if score <= 2:
        return "Low"
    elif score <= 5:
        return "Medium"
    else:
        return "High"


df["Burnout_Risk"] = df.apply(calculate_risk, axis=1)

# Save dataset
df.to_csv("student_burnout_dataset.csv", index=False)

print("Dataset created successfully!")
print("Dataset Shape:", df.shape)
print(df.head())
print("\nRisk Distribution:")
print(df["Burnout_Risk"].value_counts())
