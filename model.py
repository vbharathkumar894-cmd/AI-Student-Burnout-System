
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------
# 1. LOAD DATASET
# -----------------------------------------

df = pd.read_csv("student_burnout_dataset.csv")

print("Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# -----------------------------------------
# 2. SEPARATE FEATURES AND TARGET
# -----------------------------------------

X = df.drop("Burnout_Risk", axis=1)

y = df["Burnout_Risk"]

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Classes:")
print(y.unique())

# -----------------------------------------
# 3. TRAIN TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Records:", X_train.shape[0])
print("Testing Records:", X_test.shape[0])

# -----------------------------------------
# 4. DECISION TREE MODEL
# -----------------------------------------

dt_model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

dt_model.fit(X_train, y_train)

dt_predictions = dt_model.predict(X_test)

# -----------------------------------------
# 5. RANDOM FOREST MODEL
# -----------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# -----------------------------------------
# 6. MODEL EVALUATION
# -----------------------------------------

def evaluate_model(name, y_true, predictions):

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true, predictions, average="macro", zero_division=0
    )

    recall = recall_score(
        y_true, predictions, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_true, predictions, average="macro", zero_division=0
    )

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F1 Score:", round(f1 * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        predictions,
        zero_division=0
    ))

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


dt_results = evaluate_model(
    "Decision Tree Classifier",
    y_test,
    dt_predictions
)

rf_results = evaluate_model(
    "Random Forest Classifier",
    y_test,
    rf_predictions
)

# -----------------------------------------
# 7. COMPARE MODELS
# -----------------------------------------

results = pd.DataFrame([dt_results, rf_results])

print("\nMODEL COMPARISON")
print(results.to_string(index=False))

# -----------------------------------------
# 8. VISUALIZE MODEL COMPARISON
# -----------------------------------------

results.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(kind="bar", figsize=(10, 6))

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# -----------------------------------------
# 9. CONFUSION MATRIX
# -----------------------------------------

best_index = results["F1 Score"].idxmax()

if best_index == 0:
    best_model = dt_model
    best_predictions = dt_predictions
    best_name = "Decision Tree"
else:
    best_model = rf_model
    best_predictions = rf_predictions
    best_name = "Random Forest"

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["Low", "Medium", "High"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Low", "Medium", "High"],
    yticklabels=["Low", "Medium", "High"]
)

plt.title(f"{best_name} - Confusion Matrix")
plt.xlabel("Predicted Risk")
plt.ylabel("Actual Risk")
plt.tight_layout()
plt.show()

# -----------------------------------------
# 10. FEATURE IMPORTANCE
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE:")
print(importance.to_string(index=False))

plt.figure(figsize=(9, 5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title(f"{best_name} - Feature Importance")
plt.tight_layout()
plt.show()

# -----------------------------------------
# 11. SAVE BEST MODEL
# -----------------------------------------

joblib.dump(best_model, "best_model.pkl")

joblib.dump(
    list(X.columns),
    "model_features.pkl"
)

print("\nBest Model:", best_name)
print("Model saved successfully!")
print("Feature list saved successfully!")

print("\nMACHINE LEARNING PHASE COMPLETED!")
