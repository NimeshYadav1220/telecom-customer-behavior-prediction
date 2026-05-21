"""
Customer Behavior Prediction Model for Telecom Company
========================================================
This script trains machine learning models to predict various customer behaviors:
1. Customer Churn
2. Plan Purchase
3. Recharge Behavior
4. Complaint Likelihood
5. Data Usage Pattern
6. Plan Upgrade/Downgrade
7. Inactivity Risk
8. Offer Response
9. High Value Customer
10. Support Contact

Author: Telecom Analytics Team
Date: 2024
"""

# ==================== STEP 1: IMPORT LIBRARIES ====================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

print("="*60)
print("TELECOM CUSTOMER BEHAVIOR PREDICTION MODEL")
print("="*60)

# ==================== STEP 2: GENERATE SYNTHETIC DATASET ====================

print("\n[STEP 1] Generating synthetic telecom customer dataset...")

# Set random seed for reproducibility
np.random.seed(42)

# Number of customers
n_customers = 5000

# -------------------- A. Customer Profile Features --------------------
customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
age = np.random.randint(18, 70, n_customers)
gender = np.random.choice(['Male', 'Female'], n_customers)
location = np.random.choice(['Urban', 'Suburban', 'Rural'], n_customers, p=[0.5, 0.3, 0.2])
customer_type = np.random.choice(['New', 'Regular', 'Premium'], n_customers, p=[0.3, 0.5, 0.2])
tenure_months = np.random.randint(1, 60, n_customers)
occupation = np.random.choice(['Student', 'Professional', 'Business', 'Retired', 'Homemaker'], n_customers)
income_group = np.random.choice(['Low', 'Medium', 'High'], n_customers, p=[0.3, 0.5, 0.2])
device_type = np.random.choice(['Basic Phone', 'Smartphone 4G', 'Smartphone 5G'], n_customers, p=[0.1, 0.6, 0.3])
sim_type = np.random.choice(['Physical SIM', 'eSIM'], n_customers, p=[0.7, 0.3])

# -------------------- B. Service Usage Features --------------------
monthly_call_minutes = np.random.normal(500, 200, n_customers)
monthly_call_minutes = np.clip(monthly_call_minutes, 0, 1500)
number_of_calls = np.random.normal(300, 100, n_customers)
number_of_calls = np.clip(number_of_calls, 0, 800)
sms_count = np.random.normal(100, 80, n_customers)
sms_count = np.clip(sms_count, 0, 500)
monthly_data_usage_gb = np.random.normal(15, 10, n_customers)
monthly_data_usage_gb = np.clip(monthly_data_usage_gb, 0, 100)
average_daily_data_usage_gb = monthly_data_usage_gb / 30
roaming_usage = np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
international_calls = np.random.choice([0, 1], n_customers, p=[0.75, 0.25])
night_usage_minutes = np.random.normal(100, 80, n_customers)
night_usage_minutes = np.clip(night_usage_minutes, 0, 500)
weekend_usage_minutes = np.random.normal(200, 100, n_customers)
weekend_usage_minutes = np.clip(weekend_usage_minutes, 0, 700)

# -------------------- C. Billing and Recharge Features --------------------
monthly_bill = np.random.normal(800, 400, n_customers)
monthly_bill = np.clip(monthly_bill, 100, 3000)
recharge_amount = np.random.normal(500, 300, n_customers)
recharge_amount = np.clip(recharge_amount, 50, 2000)
recharge_frequency = np.random.poisson(3, n_customers)
recharge_frequency = np.clip(recharge_frequency, 1, 10)
last_recharge_days_ago = np.random.randint(1, 30, n_customers)
payment_method = np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'Net Banking', 'Cash Card'], n_customers)
late_payment_count = np.random.poisson(0.5, n_customers)
late_payment_count = np.clip(late_payment_count, 0, 5)
outstanding_balance = np.random.normal(0, 200, n_customers)
outstanding_balance = np.clip(outstanding_balance, 0, 1000)
auto_payment_status = np.random.choice([0, 1], n_customers, p=[0.6, 0.4])
average_revenue_per_user = monthly_bill

# -------------------- D. Plan and Subscription Features --------------------
current_plan = np.random.choice(['Basic', 'Standard', 'Premium', 'Family'], n_customers, p=[0.4, 0.3, 0.2, 0.1])
plan_price = np.where(current_plan == 'Basic', 299, 
                     np.where(current_plan == 'Standard', 499,
                              np.where(current_plan == 'Premium', 799, 999)))
plan_validity_days = np.random.choice([28, 56, 84], n_customers)
data_limit_gb = np.where(current_plan == 'Basic', 10,
                        np.where(current_plan == 'Standard', 30,
                                 np.where(current_plan == 'Premium', 75, 100)))
call_limit_minutes = np.where(current_plan == 'Basic', 500,
                             np.where(current_plan == 'Standard', 1000,
                                      np.where(current_plan == 'Premium', 2000, 3000)))
sms_limit = np.where(current_plan == 'Basic', 100,
                    np.where(current_plan == 'Standard', 300,
                             np.where(current_plan == 'Premium', 500, 800)))
plan_change_count = np.random.poisson(0.5, n_customers)
plan_change_count = np.clip(plan_change_count, 0, 4)
add_on_pack_count = np.random.poisson(1, n_customers)
add_on_pack_count = np.clip(add_on_pack_count, 0, 5)
ott_subscription = np.random.choice([0, 1], n_customers, p=[0.6, 0.4])

# -------------------- E. Complaint and Support Features --------------------
complaints_count = np.random.poisson(0.8, n_customers)
complaints_count = np.clip(complaints_count, 0, 8)
complaint_type = np.random.choice(['Network', 'Billing', 'Plan', 'Device', 'None'], n_customers, 
                                  p=[0.2, 0.2, 0.1, 0.05, 0.45])
complaint_resolution_time_hours = np.where(complaints_count > 0, np.random.exponential(24, n_customers), 0)
unresolved_complaints = np.random.poisson(0.3, n_customers)
unresolved_complaints = np.clip(unresolved_complaints, 0, complaints_count)
customer_rating = np.random.choice([1, 2, 3, 4, 5], n_customers, p=[0.05, 0.1, 0.2, 0.35, 0.3])
support_calls_count = np.random.poisson(1, n_customers)
support_calls_count = np.clip(support_calls_count, 0, 10)
chat_support_used = np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
escalated_complaints = np.random.poisson(0.2, n_customers)
escalated_complaints = np.clip(escalated_complaints, 0, complaints_count)

# -------------------- F. Network Quality Features --------------------
signal_strength = np.random.randint(1, 5, n_customers)
call_drop_count = np.random.poisson(1, n_customers)
call_drop_count = np.clip(call_drop_count, 0, 15)
internet_speed_mbps = np.random.normal(30, 15, n_customers)
internet_speed_mbps = np.clip(internet_speed_mbps, 5, 100)
network_downtime_hours = np.random.exponential(1, n_customers)
network_downtime_hours = np.clip(network_downtime_hours, 0, 10)
network_type = np.random.choice(['3G', '4G', '5G'], n_customers, p=[0.1, 0.6, 0.3])
failed_call_attempts = np.random.poisson(0.8, n_customers)
failed_call_attempts = np.clip(failed_call_attempts, 0, 10)
data_session_failures = np.random.poisson(0.6, n_customers)
data_session_failures = np.clip(data_session_failures, 0, 8)

# -------------------- G. Customer Engagement Features --------------------
app_login_count = np.random.poisson(5, n_customers)
app_login_count = np.clip(app_login_count, 0, 30)
website_visit_count = np.random.poisson(3, n_customers)
website_visit_count = np.clip(website_visit_count, 0, 20)
offer_click_count = np.random.poisson(2, n_customers)
offer_click_count = np.clip(offer_click_count, 0, 15)
campaign_response = np.random.choice([0, 1], n_customers, p=[0.7, 0.3])
notification_open_rate = np.random.uniform(0, 1, n_customers)
loyalty_points = np.random.poisson(200, n_customers)
referral_count = np.random.poisson(0.5, n_customers)
feedback_submitted = np.random.choice([0, 1], n_customers, p=[0.8, 0.2])

# -------------------- H. Churn and Risk Features --------------------
inactive_days = np.random.poisson(5, n_customers)
inactive_days = np.clip(inactive_days, 0, 60)
sim_port_request = np.random.choice([0, 1], n_customers, p=[0.92, 0.08])
competitor_offer_interest = np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
usage_drop_percentage = np.random.uniform(0, 50, n_customers)
previous_churn_history = np.random.choice([0, 1], n_customers, p=[0.95, 0.05])
contract_end_days = np.random.randint(1, 365, n_customers)

# ==================== CREATE TARGET VARIABLES WITH REALISTIC LOGIC ====================

print("\n[STEP 2] Creating target variables with realistic business logic...")

# Target 1: Will Churn (based on complaints, call drops, inactivity, etc.)
churn_score = (
    (complaints_count > 2) * 0.3 +
    (call_drop_count > 5) * 0.2 +
    (internet_speed_mbps < 15) * 0.15 +
    (inactive_days > 15) * 0.2 +
    (recharge_frequency < 2) * 0.1 +
    (customer_rating < 3) * 0.15 +
    (sim_port_request == 1) * 0.3 +
    (unresolved_complaints > 0) * 0.2
)
Will_Churn = (churn_score + np.random.uniform(0, 0.2, n_customers) > 0.45).astype(int)

# Target 2: Will Buy New Plan
buy_plan_score = (
    (monthly_data_usage_gb > data_limit_gb * 0.8) * 0.3 +
    (offer_click_count > 3) * 0.2 +
    (campaign_response == 1) * 0.2 +
    (add_on_pack_count > 2) * 0.15 +
    (notification_open_rate > 0.6) * 0.15
)
Will_Buy_New_Plan = (buy_plan_score + np.random.uniform(0, 0.2, n_customers) > 0.4).astype(int)

# Target 3: Will Recharge Again
recharge_score = (
    (recharge_frequency > 3) * 0.3 +
    (last_recharge_days_ago < 7) * 0.25 +
    (inactive_days < 5) * 0.2 +
    (monthly_data_usage_gb > 10) * 0.15 +
    (customer_rating > 3) * 0.1
)
Will_Recharge_Again = (recharge_score + np.random.uniform(0, 0.15, n_customers) > 0.4).astype(int)

# Target 4: Will Complain
complain_score = (
    (network_downtime_hours > 2) * 0.25 +
    (call_drop_count > 4) * 0.2 +
    (internet_speed_mbps < 10) * 0.2 +
    (unresolved_complaints > 0) * 0.2 +
    (failed_call_attempts > 3) * 0.15
)
Will_Complain = (complain_score + np.random.uniform(0, 0.2, n_customers) > 0.35).astype(int)

# Target 5: Will Use More Data
data_usage_score = (
    (monthly_data_usage_gb > 20) * 0.25 +
    (device_type == 'Smartphone 5G') * 0.2 +
    (ott_subscription == 1) * 0.2 +
    (app_login_count > 10) * 0.15 +
    (age < 35) * 0.1 +
    (current_plan == 'Premium') * 0.1
)
Will_Use_More_Data = (data_usage_score + np.random.uniform(0, 0.15, n_customers) > 0.4).astype(int)

# Target 6: Will Upgrade Plan
upgrade_score = (
    (monthly_data_usage_gb > data_limit_gb * 0.9) * 0.3 +
    (add_on_pack_count > 2) * 0.2 +
    (monthly_bill > plan_price * 1.2) * 0.2 +
    (income_group == 'High') * 0.15 +
    (offer_click_count > 2) * 0.15
)
Will_Upgrade_Plan = (upgrade_score + np.random.uniform(0, 0.15, n_customers) > 0.4).astype(int)

# Target 7: Will Become Inactive
inactive_score = (
    (inactive_days > 14) * 0.3 +
    (last_recharge_days_ago > 20) * 0.25 +
    (app_login_count < 2) * 0.2 +
    (monthly_data_usage_gb < 2) * 0.15 +
    (customer_rating < 3) * 0.1
)
Will_Become_Inactive = (inactive_score + np.random.uniform(0, 0.15, n_customers) > 0.4).astype(int)

# Target 8: Will Respond To Offer
offer_response_score = (
    (campaign_response == 1) * 0.3 +
    (offer_click_count > 2) * 0.25 +
    (notification_open_rate > 0.7) * 0.2 +
    (app_login_count > 5) * 0.15 +
    (customer_rating > 3) * 0.1
)
Will_Respond_To_Offer = (offer_response_score + np.random.uniform(0, 0.15, n_customers) > 0.4).astype(int)

# Target 9: Will Be High Value Customer
high_value_score = (
    (monthly_bill > 1500) * 0.25 +
    (income_group == 'High') * 0.2 +
    (add_on_pack_count > 2) * 0.2 +
    (current_plan == 'Premium') * 0.15 +
    (referral_count > 2) * 0.1 +
    (loyalty_points > 500) * 0.1
)
Will_Be_High_Value_Customer = (high_value_score + np.random.uniform(0, 0.15, n_customers) > 0.4).astype(int)

# Target 10: Will Contact Support
support_contact_score = (
    (complaints_count > 1) * 0.3 +
    (call_drop_count > 5) * 0.2 +
    (internet_speed_mbps < 15) * 0.2 +
    (support_calls_count > 2) * 0.15 +
    (chat_support_used == 1) * 0.15
)
Will_Contact_Support = (support_contact_score + np.random.uniform(0, 0.2, n_customers) > 0.35).astype(int)

# ==================== CREATE COMPLETE DATAFRAME ====================

print("\n[STEP 3] Creating complete dataset...")

data = pd.DataFrame({
    # Customer Profile
    'Customer_ID': customer_ids,
    'Age': age,
    'Gender': gender,
    'Location': location,
    'Customer_Type': customer_type,
    'Tenure_Months': tenure_months,
    'Occupation': occupation,
    'Income_Group': income_group,
    'Device_Type': device_type,
    'SIM_Type': sim_type,
    
    # Service Usage
    'Monthly_Call_Minutes': monthly_call_minutes,
    'Number_of_Calls': number_of_calls,
    'SMS_Count': sms_count,
    'Monthly_Data_Usage_GB': monthly_data_usage_gb,
    'Average_Daily_Data_Usage_GB': average_daily_data_usage_gb,
    'Roaming_Usage': roaming_usage,
    'International_Calls': international_calls,
    'Night_Usage_Minutes': night_usage_minutes,
    'Weekend_Usage_Minutes': weekend_usage_minutes,
    
    # Billing and Recharge
    'Monthly_Bill': monthly_bill,
    'Recharge_Amount': recharge_amount,
    'Recharge_Frequency': recharge_frequency,
    'Last_Recharge_Days_Ago': last_recharge_days_ago,
    'Payment_Method': payment_method,
    'Late_Payment_Count': late_payment_count,
    'Outstanding_Balance': outstanding_balance,
    'Auto_Payment_Status': auto_payment_status,
    'Average_Revenue_Per_User': average_revenue_per_user,
    
    # Plan and Subscription
    'Current_Plan': current_plan,
    'Plan_Price': plan_price,
    'Plan_Validity_Days': plan_validity_days,
    'Data_Limit_GB': data_limit_gb,
    'Call_Limit_Minutes': call_limit_minutes,
    'SMS_Limit': sms_limit,
    'Plan_Change_Count': plan_change_count,
    'Add_On_Pack_Count': add_on_pack_count,
    'OTT_Subscription': ott_subscription,
    
    # Complaint and Support
    'Complaints_Count': complaints_count,
    'Complaint_Type': complaint_type,
    'Complaint_Resolution_Time_Hours': complaint_resolution_time_hours,
    'Unresolved_Complaints': unresolved_complaints,
    'Customer_Rating': customer_rating,
    'Support_Calls_Count': support_calls_count,
    'Chat_Support_Used': chat_support_used,
    'Escalated_Complaints': escalated_complaints,
    
    # Network Quality
    'Signal_Strength': signal_strength,
    'Call_Drop_Count': call_drop_count,
    'Internet_Speed_Mbps': internet_speed_mbps,
    'Network_Downtime_Hours': network_downtime_hours,
    'Network_Type': network_type,
    'Failed_Call_Attempts': failed_call_attempts,
    'Data_Session_Failures': data_session_failures,
    
    # Customer Engagement
    'App_Login_Count': app_login_count,
    'Website_Visit_Count': website_visit_count,
    'Offer_Click_Count': offer_click_count,
    'Campaign_Response': campaign_response,
    'Notification_Open_Rate': notification_open_rate,
    'Loyalty_Points': loyalty_points,
    'Referral_Count': referral_count,
    'Feedback_Submitted': feedback_submitted,
    
    # Churn and Risk
    'Inactive_Days': inactive_days,
    'SIM_Port_Request': sim_port_request,
    'Competitor_Offer_Interest': competitor_offer_interest,
    'Usage_Drop_Percentage': usage_drop_percentage,
    'Previous_Churn_History': previous_churn_history,
    'Contract_End_Days': contract_end_days,
    
    # Target Variables
    'Will_Churn': Will_Churn,
    'Will_Buy_New_Plan': Will_Buy_New_Plan,
    'Will_Recharge_Again': Will_Recharge_Again,
    'Will_Complain': Will_Complain,
    'Will_Use_More_Data': Will_Use_More_Data,
    'Will_Upgrade_Plan': Will_Upgrade_Plan,
    'Will_Become_Inactive': Will_Become_Inactive,
    'Will_Respond_To_Offer': Will_Respond_To_Offer,
    'Will_Be_High_Value_Customer': Will_Be_High_Value_Customer,
    'Will_Contact_Support': Will_Contact_Support
})

print(f"Dataset created with {data.shape[0]} rows and {data.shape[1]} columns")

# Save dataset
import os
os.makedirs('data', exist_ok=True)
data.to_csv('data/telecom_customer_behavior.csv', index=False)
print("Dataset saved to 'data/telecom_customer_behavior.csv'")

# ==================== STEP 3: DATA EXPLORATION ====================

print("\n[STEP 4] Data Exploration...")
print("\nFirst 5 rows:")
print(data.head())
print("\nDataset shape:", data.shape)
print("\nColumn names:")
print(data.columns.tolist())
print("\nData types:")
print(data.dtypes.value_counts())
print("\nMissing values:")
print(data.isnull().sum().sum())
print("\nSummary statistics for numerical columns:")
print(data.describe())

# Target variable distribution
print("\nTarget variable distribution:")
target_cols = ['Will_Churn', 'Will_Buy_New_Plan', 'Will_Recharge_Again', 'Will_Complain',
               'Will_Use_More_Data', 'Will_Upgrade_Plan', 'Will_Become_Inactive',
               'Will_Respond_To_Offer', 'Will_Be_High_Value_Customer', 'Will_Contact_Support']

for col in target_cols:
    print(f"{col}: {data[col].value_counts().to_dict()}")

# ==================== STEP 4: DATA PREPROCESSING ====================

print("\n[STEP 5] Data Preprocessing...")

# Remove duplicate records
initial_rows = len(data)
data = data.drop_duplicates()
print(f"Removed {initial_rows - len(data)} duplicate records")

# Separate features and targets
feature_cols = [col for col in data.columns if col not in target_cols and col != 'Customer_ID']
X = data[feature_cols]
y = data[target_cols]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("Categorical encoding and numerical scaling completed")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y[target_cols[0]])
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ==================== STEP 5: FEATURE ENGINEERING ====================

print("\n[STEP 6] Feature Engineering...")

# Add engineered features to training and test sets
def add_engineered_features(X_df):
    X_df = X_df.copy()
    # Average monthly usage
    X_df['Avg_Monthly_Usage'] = (X_df['Monthly_Call_Minutes'] + X_df['Monthly_Data_Usage_GB'] * 100) / 2
    # Data usage ratio (usage/limit)
    X_df['Data_Usage_Ratio'] = X_df['Monthly_Data_Usage_GB'] / (X_df['Data_Limit_GB'] + 1)
    # Complaint rate
    X_df['Complaint_Rate'] = X_df['Complaints_Count'] / (X_df['Tenure_Months'] + 1)
    # Recharge regularity score
    X_df['Recharge_Regularity'] = X_df['Recharge_Frequency'] / (X_df['Last_Recharge_Days_Ago'] + 1)
    # Network issue score
    X_df['Network_Issue_Score'] = (X_df['Call_Drop_Count'] + X_df['Data_Session_Failures']) / 10
    # Customer value score
    X_df['Customer_Value_Score'] = X_df['Monthly_Bill'] * X_df['Tenure_Months'] / 1000
    # Engagement score
    X_df['Engagement_Score'] = (X_df['App_Login_Count'] + X_df['Website_Visit_Count'] + X_df['Offer_Click_Count']) / 30
    # Churn risk score
    X_df['Churn_Risk_Score'] = (X_df['Complaints_Count'] * 0.3 + X_df['Call_Drop_Count'] * 0.2 + 
                                X_df['Inactive_Days'] * 0.3 + X_df['Late_Payment_Count'] * 0.2)
    return X_df

X_train = add_engineered_features(X_train)
X_test = add_engineered_features(X_test)

print(f"Added 8 engineered features. New feature count: {X_train.shape[1]}")

# ==================== STEP 6: MODEL TRAINING ====================

print("\n[STEP 7] Training Models...")

# Define models
models = {
    'Logistic Regression': MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=42)),
    'Decision Tree': MultiOutputClassifier(DecisionTreeClassifier(random_state=42)),
    'Random Forest': MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
}

results = {}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics for each target
    metrics = {}
    for i, target in enumerate(target_cols):
        acc = accuracy_score(y_test[target], y_pred[:, i])
        prec = precision_score(y_test[target], y_pred[:, i], average='binary', zero_division=0)
        rec = recall_score(y_test[target], y_pred[:, i], average='binary', zero_division=0)
        f1 = f1_score(y_test[target], y_pred[:, i], average='binary', zero_division=0)
        metrics[target] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1}
    
    # Calculate average metrics
    avg_acc = np.mean([metrics[t]['Accuracy'] for t in target_cols])
    avg_f1 = np.mean([metrics[t]['F1-Score'] for t in target_cols])
    
    results[name] = {
        'model': model,
        'metrics': metrics,
        'avg_accuracy': avg_acc,
        'avg_f1': avg_f1,
        'predictions': y_pred
    }
    
    print(f"{name} - Average Accuracy: {avg_acc:.4f}, Average F1-Score: {avg_f1:.4f}")

# ==================== STEP 7: MODEL EVALUATION DETAILS ====================

print("\n[STEP 8] Detailed Model Evaluation...")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['avg_f1'])
best_model = results[best_model_name]['model']
print(f"\nBest Model: {best_model_name} (Avg F1-Score: {results[best_model_name]['avg_f1']:.4f})")

# Display detailed metrics for best model
print("\nDetailed Metrics for Best Model:")
print("="*80)
for target in target_cols:
    metrics = results[best_model_name]['metrics'][target]
    print(f"\n{target}:")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")

# Confusion matrix for first target
print("\nConfusion Matrix for Churn Prediction (Best Model):")
cm = confusion_matrix(y_test['Will_Churn'], results[best_model_name]['predictions'][:, 0])
print(cm)

# ==================== STEP 8: SAVE BEST MODEL AND PREPROCESSORS ====================

print("\n[STEP 9] Saving best model and preprocessors...")

# Create model directory
os.makedirs('model', exist_ok=True)

# Save model
joblib.dump(best_model, 'model/best_model.pkl')
print("Best model saved to 'model/best_model.pkl'")

# Save scaler
joblib.dump(scaler, 'model/scaler.pkl')
print("Scaler saved to 'model/scaler.pkl'")

# Save label encoders
joblib.dump(label_encoders, 'model/encoder.pkl')
print("Label encoders saved to 'model/encoder.pkl'")

# Save feature columns list
joblib.dump(feature_cols, 'model/feature_columns.pkl')
joblib.dump(target_cols, 'model/target_columns.pkl')
joblib.dump(categorical_cols, 'model/categorical_cols.pkl')
joblib.dump(numerical_cols, 'model/numerical_cols.pkl')

print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"\nFinal Model Performance:")
print(f"Model: {best_model_name}")
print(f"Average Accuracy: {results[best_model_name]['avg_accuracy']:.4f}")
print(f"Average F1-Score: {results[best_model_name]['avg_f1']:.4f}")
print("\nYou can now run the Streamlit app using: streamlit run app.py")