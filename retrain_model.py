import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

print("="*60)
print("RETRAINING MODEL WITH PROPER FEATURE SAVING")
print("="*60)

# Load the existing dataset
print("\nLoading dataset...")
df = pd.read_csv('data/telecom_customer_behavior.csv')
print(f"Dataset shape: {df.shape}")

# Define target columns
target_cols = ['Will_Churn', 'Will_Buy_New_Plan', 'Will_Recharge_Again', 'Will_Complain',
               'Will_Use_More_Data', 'Will_Upgrade_Plan', 'Will_Become_Inactive',
               'Will_Respond_To_Offer', 'Will_Be_High_Value_Customer', 'Will_Contact_Support']

# Define feature columns (exclude Customer_ID and targets)
feature_cols = [col for col in df.columns if col not in target_cols and col != 'Customer_ID']

print(f"Number of features: {len(feature_cols)}")

# Prepare X and y
X = df[feature_cols].copy()
y = df[target_cols]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded: {col}")

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("\nTraining Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model = MultiOutputClassifier(rf, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.4f}")

# Save everything
print("\nSaving models and preprocessors...")
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/best_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(label_encoders, 'model/encoder.pkl')
joblib.dump(feature_cols, 'model/feature_columns.pkl')
joblib.dump(target_cols, 'model/target_columns.pkl')
joblib.dump(categorical_cols, 'model/categorical_cols.pkl')
joblib.dump(numerical_cols, 'model/numerical_cols.pkl')

print("\n✅ ALL FILES SAVED SUCCESSFULLY!")
print("="*60)
