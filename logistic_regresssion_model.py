import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# --- Logistic Regression Model ---
# Description: LogisticRegression() class from scikit-learn is initialized and trained 
# using fit(). After training, predict() generates predictions on unseen test data.
# Model evaluated using confusion matrix and classification report: precision, recall, F1-score.
# Logistic Regression is fast, easy to interpret, and works for binary classification.

# 1. Load the preprocessed dataset from your last step
df = pd.read_csv('final_dataset.csv')
print("1. Dataset loaded:", df.shape)

# 2. Split features X and target y
# Replace 'CREDIT_TARGET' with your actual target column name
X = df.drop('CREDIT_TARGET', axis=1) 
y = df['CREDIT_TARGET']

# 3. Handle any remaining NaN values - LogisticRegression can't handle NaN
X = X.fillna(0)

# 4. Train-test split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("2. Train size:", X_train.shape, "Test size:", X_test.shape)

# 5. Feature scaling - Important for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Initialize LogisticRegression() class
model = LogisticRegression(max_iter=1000, random_state=42)

# 7. Train using fit() method
model.fit(X_train_scaled, y_train)
print("3. Model trained using fit()")

# 8. Generate predictions using predict() method on unseen test data
y_pred = model.predict(X_test_scaled)
print("4. Predictions generated using predict()")

# 9. Evaluate model performance - compare predicted vs actual
accuracy = accuracy_score(y_test, y_pred)
print(f"5. Accuracy: {accuracy:.4f}")

# 10. Confusion Matrix
print("\n6. Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Format: [[True Negatives, False Positives],
#          [False Negatives, True Positives]]

# 11. Classification Report: precision, recall, F1-score, support
print("\n7. Classification Report:")
print(classification_report(y_test, y_pred))