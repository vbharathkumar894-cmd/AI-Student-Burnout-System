
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# 1. Load Dataset
# --------------------------------

df = pd.read_csv("student_burnout_dataset.csv")

print("DATASET LOADED SUCCESSFULLY!")
print("-" * 50)

# --------------------------------
# 2. Basic Dataset Information
# --------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# --------------------------------
# 3. Risk Distribution
# --------------------------------

print("\nRisk Distribution:")
print(df["Burnout_Risk"].value_counts())

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Burnout_Risk",
    order=["Low", "Medium", "High"]
)

plt.title("Student Academic Strain Risk Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.show()

# --------------------------------
# 4. Study Hours vs Risk
# --------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Burnout_Risk",
    y="Study_Hours",
    order=["Low", "Medium", "High"]
)

plt.title("Study Hours vs Academic Strain Risk")
plt.xlabel("Risk Category")
plt.ylabel("Study Hours")

plt.tight_layout()
plt.show()

# --------------------------------
# 5. Sleep Hours vs Risk
# --------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Burnout_Risk",
    y="Sleep_Hours",
    order=["Low", "Medium", "High"]
)

plt.title("Sleep Hours vs Academic Strain Risk")
plt.xlabel("Risk Category")
plt.ylabel("Sleep Hours")

plt.tight_layout()
plt.show()

# --------------------------------
# 6. Stress Level vs Risk
# --------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Burnout_Risk",
    y="Stress_Level",
    order=["Low", "Medium", "High"]
)

plt.title("Stress Level vs Academic Strain Risk")
plt.xlabel("Risk Category")
plt.ylabel("Stress Level")

plt.tight_layout()
plt.show()

# --------------------------------
# 7. Correlation Heatmap
# --------------------------------

plt.figure(figsize=(10, 6))

correlation = df.select_dtypes(
    include="number"
).corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Student Feature Correlation")

plt.tight_layout()
plt.show()

# --------------------------------
# 8. Final EDA Summary
# --------------------------------

print("\nEDA COMPLETED SUCCESSFULLY!")

print("\nKey Observations:")

print("1. Average Study Hours:",
      round(df["Study_Hours"].mean(), 2))

print("2. Average Sleep Hours:",
      round(df["Sleep_Hours"].mean(), 2))

print("3. Average Attendance:",
      round(df["Attendance"].mean(), 2))

print("4. Average Stress Level:",
      round(df["Stress_Level"].mean(), 2))

print("\nAll EDA visualizations generated!")
