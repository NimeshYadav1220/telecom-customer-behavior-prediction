import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telecom Predictor", page_icon="📱", layout="wide")

st.title("📱 Telecom Customer Behavior Prediction System")
st.markdown("---")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('model/best_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    encoders = joblib.load('model/encoder.pkl')
    feature_cols = joblib.load('model/feature_columns.pkl')
    target_cols = joblib.load('model/target_columns.pkl')
    cat_cols = joblib.load('model/categorical_cols.pkl')
    num_cols = joblib.load('model/numerical_cols.pkl')
    return model, scaler, encoders, feature_cols, target_cols, cat_cols, num_cols

model, scaler, encoders, feature_cols, target_cols, cat_cols, num_cols = load_models()
st.success("✅ Models loaded successfully!")

# Sidebar inputs
st.sidebar.header("📝 Enter Customer Details")

with st.sidebar.form("prediction_form"):
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure (Months)", 1, 120, 24)
    call_minutes = st.number_input("Monthly Call Minutes", 0, 3000, 800)
    data_gb = st.number_input("Monthly Data (GB)", 0.0, 200.0, 25.0)
    monthly_bill = st.number_input("Monthly Bill (₹)", 100, 5000, 1200)
    recharge_freq = st.number_input("Recharge Frequency", 1, 10, 4)
    complaints = st.number_input("Complaints Count", 0, 20, 0)
    call_drops = st.number_input("Call Drops", 0, 20, 1)
    rating = st.slider("Customer Rating", 1, 5, 5)
    app_logins = st.number_input("App Logins", 0, 100, 15)
    
    submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

# Target labels
target_labels = {
    'Will_Churn': '🚪 Leave company (Churn)',
    'Will_Buy_New_Plan': '🛒 Buy new plan',
    'Will_Recharge_Again': '🔄 Recharge again',
    'Will_Complain': '😤 Complain',
    'Will_Use_More_Data': '📊 Use more data',
    'Will_Upgrade_Plan': '⬆️ Upgrade plan',
    'Will_Become_Inactive': '💤 Become inactive',
    'Will_Respond_To_Offer': '🎯 Respond to offers',
    'Will_Be_High_Value_Customer': '⭐ Become high-value',
    'Will_Contact_Support': '📞 Contact support'
}

if submitted:
    with st.spinner("Predicting..."):
        # Create input row
        input_data = {
            'Age': age,
            'Gender': 'Male',
            'Location': 'Urban',
            'Customer_Type': 'Regular',
            'Tenure_Months': tenure,
            'Occupation': 'Professional',
            'Income_Group': 'Medium',
            'Device_Type': 'Smartphone 4G',
            'SIM_Type': 'Physical SIM',
            'Monthly_Call_Minutes': call_minutes,
            'Number_of_Calls': call_minutes/2,
            'SMS_Count': 100,
            'Monthly_Data_Usage_GB': data_gb,
            'Average_Daily_Data_Usage_GB': data_gb/30,
            'Roaming_Usage': 0,
            'International_Calls': 0,
            'Night_Usage_Minutes': 100,
            'Weekend_Usage_Minutes': 200,
            'Monthly_Bill': monthly_bill,
            'Recharge_Amount': monthly_bill*0.6,
            'Recharge_Frequency': recharge_freq,
            'Last_Recharge_Days_Ago': 7,
            'Payment_Method': 'UPI',
            'Late_Payment_Count': 0,
            'Outstanding_Balance': 0,
            'Auto_Payment_Status': 0,
            'Average_Revenue_Per_User': monthly_bill,
            'Current_Plan': 'Standard',
            'Plan_Price': 499,
            'Plan_Validity_Days': 28,
            'Data_Limit_GB': 30,
            'Call_Limit_Minutes': 1000,
            'SMS_Limit': 300,
            'Plan_Change_Count': 0,
            'Add_On_Pack_Count': 1,
            'OTT_Subscription': 0,
            'Complaints_Count': complaints,
            'Complaint_Type': 'None',
            'Complaint_Resolution_Time_Hours': 0,
            'Unresolved_Complaints': 0,
            'Customer_Rating': rating,
            'Support_Calls_Count': complaints + 1,
            'Chat_Support_Used': 0,
            'Escalated_Complaints': 0,
            'Signal_Strength': 4,
            'Call_Drop_Count': call_drops,
            'Internet_Speed_Mbps': 30,
            'Network_Downtime_Hours': 1,
            'Network_Type': '4G',
            'Failed_Call_Attempts': call_drops,
            'Data_Session_Failures': call_drops,
            'App_Login_Count': app_logins,
            'Website_Visit_Count': 5,
            'Offer_Click_Count': 3,
            'Campaign_Response': 0,
            'Notification_Open_Rate': 0.5,
            'Loyalty_Points': 200,
            'Referral_Count': 1,
            'Feedback_Submitted': 0,
            'Inactive_Days': 5,
            'SIM_Port_Request': 0,
            'Competitor_Offer_Interest': 0,
            'Usage_Drop_Percentage': 10,
            'Previous_Churn_History': 0,
            'Contract_End_Days': 180,
        }
        
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Add engineered features
        df['Avg_Monthly_Usage'] = (df['Monthly_Call_Minutes'] + df['Monthly_Data_Usage_GB'] * 100) / 2
        df['Data_Usage_Ratio'] = df['Monthly_Data_Usage_GB'] / (df['Data_Limit_GB'] + 1)
        df['Complaint_Rate'] = df['Complaints_Count'] / (df['Tenure_Months'] + 1)
        df['Recharge_Regularity'] = df['Recharge_Frequency'] / (df['Last_Recharge_Days_Ago'] + 1)
        df['Network_Issue_Score'] = (df['Call_Drop_Count'] + df['Data_Session_Failures']) / 10
        df['Customer_Value_Score'] = df['Monthly_Bill'] * df['Tenure_Months'] / 1000
        df['Engagement_Score'] = (df['App_Login_Count'] + df['Website_Visit_Count'] + df['Offer_Click_Count']) / 30
        df['Churn_Risk_Score'] = (df['Complaints_Count'] * 0.3 + df['Call_Drop_Count'] * 0.2 + 
                                  df['Inactive_Days'] * 0.3 + df['Late_Payment_Count'] * 0.2)
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns
        df = df[feature_cols]
        
        # Encode categorical
        for col in cat_cols:
            if col in df.columns and col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except:
                    df[col] = 0
        
        # Scale numerical
        df[num_cols] = scaler.transform(df[num_cols])
        
        # Predict
        predictions = model.predict(df)[0]
        
        # Display results
        st.header("📊 Results")
        
        col1, col2 = st.columns(2)
        items = list(target_labels.items())
        
        with col1:
            for i, (target, label) in enumerate(items[:5]):
                result = "✅ Yes" if predictions[i] == 1 else "❌ No"
                st.write(f"{label}: {result}")
        
        with col2:
            for i, (target, label) in enumerate(items[5:]):
                result = "✅ Yes" if predictions[i+5] == 1 else "❌ No"
                st.write(f"{label}: {result}")
        
        # Risk assessment
        st.markdown("---")
        if predictions[0] == 1:
            st.error("🔴 HIGH CHURN RISK - Immediate action required!")
        else:
            st.success("🟢 LOW RISK - Customer appears stable")
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        if predictions[0] == 1:
            st.write("• Offer special discount and loyalty rewards")
            st.write("• Contact customer personally")
        if predictions[1] == 1 or predictions[5] == 1:
            st.write("• Recommend plan upgrade with discount")
        if predictions[4] == 1:
            st.write("• Suggest higher data plan")
        if predictions[8] == 1:
            st.write("• Provide VIP support and exclusive offers")

st.markdown("---")
st.caption("© 2024 Telecom Analytics")
