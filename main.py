import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)

# Imbalanced dataset
data = {
    "study_hours": [1,1,2,2,3,3,3,4,4,4,5,6],
    "result":      [0,0,0,0,0,0,0,0,0,1,1,1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

X = df[["study_hours"]]
y = df["result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Model with class_weight
model = LogisticRegression(class_weight="balanced")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fail", "Pass"]
)
disp.plot(cmap="Blues", ax=ax)

# Add evaluation text
metrics_text = (
    f"Precision: {precision:.2f}\n"
    f"Recall: {recall:.2f}\n"
    f"F1-score: {f1:.2f}"
)

plt.gcf().text(
    0.72, 0.5, metrics_text,
    fontsize=11,
    bbox=dict(boxstyle="round", facecolor="lightyellow")
)

plt.title("Student Pass/Fail – Confusion Matrix with Evaluation Metrics")

# ⭐ SAVE IMAGE (important line)
plt.savefig("confusion_matrix_evaluation.png", dpi=300, bbox_inches="tight")

plt.show()
