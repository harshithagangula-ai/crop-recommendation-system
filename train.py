import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Split features and label
X = data.drop("label", axis=1)
y = data["label"]

# Store feature names and crop classes for the app
feature_names = X.columns.tolist()
crop_classes = y.unique().tolist()

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(f"✅ Cross Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# Feature importance plot
importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, color="#2e7d32")
plt.title("Feature Importance", fontsize=14)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
print("✅ Model saved as model.pkl")

# Save metadata for app
metadata = {
    "feature_names": feature_names,
    "crop_classes": sorted(crop_classes),
    "accuracy": round(accuracy, 4),
    "cv_accuracy": round(cv_scores.mean(), 4),
    "feature_importance": dict(zip(feature_names, importances.tolist()))
}
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f)
print("✅ Metadata saved as model_metadata.json")
