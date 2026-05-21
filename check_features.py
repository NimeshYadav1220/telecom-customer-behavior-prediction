import joblib

# Load the feature columns
feature_cols = joblib.load('model/feature_columns.pkl')

print(f"Total features expected: {len(feature_cols)}")
print("\nAll feature columns:")
for i, col in enumerate(feature_cols):
    print(f"{i+1:3}. {col}")

# Check which are engineered features
engineered = ['Avg_Monthly_Usage', 'Churn_Risk_Score', 'Complaint_Rate', 
              'Customer_Value_Score', 'Data_Usage_Ratio', 'Engagement_Score',
              'Network_Issue_Score', 'Recharge_Regularity']

print("\n" + "="*50)
print("Engineered features needed:")
for eng in engineered:
    if eng in feature_cols:
        print(f"✓ {eng}")
    else:
        print(f"✗ {eng} not found")
