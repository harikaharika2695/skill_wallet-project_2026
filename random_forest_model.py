import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Random Forest Model ---
# Description: RandomForestClassifier() from scikit-learn is initialized and trained 
# using fit(). Random Forest creates multiple decision trees and combines predictions 
# using majority voting to improve accuracy and reduce overfitting. 
# After training, predict() generates predictions on unseen test data.
# Model evaluated using confusion matrix and classification report: precision, recall, F1-score.

def random_forest(X_train, X_test, y_train, y_test):
    """
    Function to train, test, and evaluate Random Forest model
    using training and testing datasets: X_train, X_test, y_train, y_test
    """
    
    # 1. Initialize RandomForestClassifier() class
    # n_estimators=100 means 100 decision trees. random_state for reproducibility.
    # class_weight='balanced' helps with imbalanced data like Home Credit
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    
    # 2. Train using fit() method
    model.fit(X_train, y_train)
    print("1. Random Forest model trained using fit()")
    
    # 3. Generate predictions using predict() method on unseen test data
    y_pred = model.predict(X_test)
    print("2. Predictions generated using predict()")
    
    # 4. Evaluate model performance - compare predicted vs actual
    accuracy = accuracy_score(y_test, y_pred)
    print(f"3. Accuracy: {accuracy:.4f}")
    
    # 5. Confusion Matrix
    print("\n4. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # 6. Classification Report: precision, recall, F1-score, support
    print("\n5. Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# --- Main execution ---
if __name__ == "__main__":
    # 1. Load the preprocessed dataset
    df = pd.read_csv('final_dataset.csv')
    print("Dataset loaded:", df.shape)
    
    # 2. Split features X and target y
    # Replace 'CREDIT_TARGET' with your actual target column name
    X = df.drop('CREDIT_TARGET', axis=1)
    y = df['CREDIT_TARGET']
    
    # 3. Handle NaN values - Random Forest can handle NaN but filling is safer
    X = X.fillna(0)
    
    # 4. Train-test split: 80% train, 20% test. stratify=y for imbalanced data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Train size:", X_train.shape, "Test size:", X_test.shape)
    
    # 5. Call the random_forest() function to train, test, evaluate
    rf_model = random_forest(X_train, X_test, y_train, y_test)