import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Telecom Predictor", page_icon="📱", layout="wide")

st.title("📱 Telecom Customer Behavior Prediction System")
st.markdown("---")

# Load all models
@st.cache_resource
def load_all_models():
    model = joblib.load('model/best_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    encoders = joblib.load('model/encoder.pkl')
    feature_cols = joblib.load('model/feature_columns.pkl')
    target_cols = joblib.load('model/target_columns.pkl')
    cat_cols = joblib.load('model/categorical_cols.pkl')
    num_cols = joblib.load('model/numerical_cols.pkl')
    return model, scaler, encoders, feature_cols, target_cols, cat_cols, num_cols

model, scaler, encoders, feature_cols, target_cols, cat_cols, num_cols = load_all_models()
st.success("✅ Models loaded successfully!")

# Print feature columns for debugging (hidden in sidebar)
with st.sidebar.expander("Debug Info"):
    st.write(f"Total features expected: {len(feature_cols)}")
    st.write("First 20 features:", feature_cols[:20])

# Sidebar inputs
st.sidebar.header("📝 Enter Customer Details")

with st.sidebar.form("input_form"):
    st.subheader("Basic Information")
    age = st.number_input("Age", 18, 100, 35)
    tenure = st.number_input("Tenure (Months)", 1, 120, 24)
    
    st.subheader("Usage Pattern")
    call_minutes = st.number_input("Monthly Call Minutes", 0, 3000, 800)
    data_gb = st.number_input("Monthly Data (GB)", 0.0, 200.0, 25.0)
    monthly_bill = st.number_input("Monthly Bill (₹)", 100, 5000, 1200)
    recharge_freq = st.number_input("Recharge Frequency (per month)", 1, 10, 4)
    
    st.subheader("Service Quality")
    complaints = st.number_input("Complaints Count", 0, 20, 0)
    call_drops = st.number_input("Call Drops", 0, 20, 1)
    rating = st.slider("Customer Rating (1-5)", 1, 5, 5)
    
    st.subheader("Engagement")
    app_logins = st.number_input("App Logins per month", 0, 100, 15)
    website_visits = st.number_input("Website Visits per month", 0, 50, 8)
    offer_clicks = st.number_input("Offer Clicks per month", 0, 50, 5)
    
    submitted = st.form_submit_button("🔮 Predict Customer Behavior", use_container_width=True)

# Function to add engineered features (MUST MATCH training)
def add_engineered_features(df):
    """Add engineered features exactly as in training"""
    df = df.copy()
    
    # These calculations must match train_model.py
    df['Avg_Monthly_Usage'] = (df['Monthly_Call_Minutes'] + df['Monthly_Data_Usage_GB'] * 100) / 2
    df['Data_Usage_Ratio'] = df['Monthly_Data_Usage_GB'] / (df['Data_Limit_GB'] + 1)
    df['Complaint_Rate'] = df['Complaints_Count'] / (df['Tenure_Months'] + 1)
    df['Recharge_Regularity'] = df['Recharge_Frequency'] / (df['Last_Recharge_Days_Ago'] + 1)
    df['Network_Issue_Score'] = (df['Call_Drop_Count'] + df['Data_Session_Failures']) / 10
    df['Customer_Value_Score'] = df['Monthly_Bill'] * df['Tenure_Months'] / 1000
    df['Engagement_Score'] = (df['App_Login_Count'] + df['Website_Visit_Count'] + df['Offer_Click_Count']) / 30
    df['Churn_Risk_Score'] = (df['Complaints_Count'] * 0.3 + df['Call_Drop_Count'] * 0.2 + 
                              df['Inactive_Days'] * 0.3 + df['Late_Payment_Count'] * 0.2)
    
    return df

# Create complete feature dictionary
def create_full_features(user_input):
    """Create all features with defaults"""
    features = {
        # Profile
        'Age': user_input['age'], 
        'Gender': 'Male', 
        'Location': 'Urban',
        'Customer_Type': 'Regular', 
        'Tenure_Months': user_input['tenure'],
        'Occupation': 'Professional', 
        'Income_Group': 'Medium',
        'Device_Type': 'Smartphone 4G', 
        'SIM_Type': 'Physical SIM',
        
        # Usage
        'Monthly_Call_Minutes': user_input['call_minutes'], 
        'Number_of_Calls': user_input['call_minutes'] / 2,
        'SMS_Count': 100, 
        'Monthly_Data_Usage_GB': user_input['data_gb'],
        'Average_Daily_Data_Usage_GB': user_input['data_gb']/30, 
        'Roaming_Usage': 0,
        'International_Calls': 0, 
        'Night_Usage_Minutes': 100, 
        'Weekend_Usage_Minutes': 200,
        
        # Billing
        'Monthly_Bill': user_input['monthly_bill'], 
        'Recharge_Amount': user_input['monthly_bill'] * 0.6,
        'Recharge_Frequency': user_input['recharge_freq'], 
        'Last_Recharge_Days_Ago': 7,
        'Payment_Method': 'UPI', 
        'Late_Payment_Count': 0, 
        'Outstanding_Balance': 0,
        'Auto_Payment_Status': 0, 
        'Average_Revenue_Per_User': user_input['monthly_bill'],
        
        # Plan
        'Current_Plan': 'Standard', 
        'Plan_Price': 499, 
        'Plan_Validity_Days': 28,
        'Data_Limit_GB': 30, 
        'Call_Limit_Minutes': 1000, 
        'SMS_Limit': 300,
        'Plan_Change_Count': 0, 
        'Add_On_Pack_Count': 1, 
        'OTT_Subscription': 0,
        
        # Complaints
        'Complaints_Count': user_input['complaints'], 
        'Complaint_Type': 'None',
        'Complaint_Resolution_Time_Hours': 0, 
        'Unresolved_Complaints': 0,
        'Customer_Rating': user_input['rating'], 
        'Support_Calls_Count': user_input['complaints'] + 1,
        'Chat_Support_Used': 0, 
        'Escalated_Complaints': 0,
        
        # Network
        'Signal_Strength': 4, 
        'Call_Drop_Count': user_input['call_drops'],
        'Internet_Speed_Mbps': 30, 
        'Network_Downtime_Hours': 1, 
        'Network_Type': '4G',
        'Failed_Call_Attempts': user_input['call_drops'], 
        'Data_Session_Failures': user_input['call_drops'],
        
        # Engagement
        'App_Login_Count': user_input['app_logins'], 
        'Website_Visit_Count': user_input['website_visits'],
        'Offer_Click_Count': user_input['offer_clicks'], 
        'Campaign_Response': 0, 
        'Notification_Open_Rate': 0.5,
        'Loyalty_Points': 200, 
        'Referral_Count': 1, 
        'Feedback_Submitted': 0,
        
        # Risk
        'Inactive_Days': 5, 
        'SIM_Port_Request': 0, 
        'Competitor_Offer_Interest': 0,
        'Usage_Drop_Percentage': 10, 
        'Previous_Churn_History': 0, 
        'Contract_End_Days': 180,
    }
    
    return features

# Target labels for display
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
    with st.spinner("Analyzing customer data..."):
        # Prepare input
        user_input = {
            'age': age, 
            'tenure': tenure, 
            'call_minutes': call_minutes,
            'data_gb': data_gb, 
            'monthly_bill': monthly_bill,
            'recharge_freq': recharge_freq, 
            'complaints': complaints,
            'call_drops': call_drops, 
            'rating': rating, 
            'app_logins': app_logins,
            'website_visits': website_visits,
            'offer_clicks': offer_clicks
        }
        
        # Create base features
        features = create_full_features(user_input)
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Add engineered features
        df = add_engineered_features(df)
        
        # Ensure all feature columns exist - fill missing with 0
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder columns to match training
        df = df[feature_cols]
        
        # Encode categorical variables
        for col in cat_cols:
            if col in df.columns and col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except Exception as e:
                    # If unknown category, use first known category
                    if hasattr(encoders[col], 'classes_'):
                        df[col] = 0
                    else:
                        df[col] = 0
        
        # Scale numerical features
        try:
            df[num_cols] = scaler.transform(df[num_cols])
        except Exception as e:
            st.error(f"Scaling error: {e}")
            st.stop()
        
        # Make predictions
        predictions = model.predict(df)[0]
        
        # Display results
        st.header("📊 Prediction Results")
        
        # Create two columns for results
        col1, col2 = st.columns(2)
        items = list(target_labels.items())
        
        with col1:
            for i, (target, label) in enumerate(items[:5]):
                result = "Yes" if predictions[i] == 1 else "No"
                if result == "Yes":
                    st.success(f"✅ {label}: **{result}**")
                else:
                    st.info(f"❌ {label}: **{result}**")
        
        with col2:
            for i, (target, label) in enumerate(items[5:]):
                result = "Yes" if predictions[i+5] == 1 else "No"
                if result == "Yes":
                    st.success(f"✅ {label}: **{result}**")
                else:
                    st.info(f"❌ {label}: **{result}**")
        
        # Risk Assessment
        st.markdown("---")
        st.subheader("⚠️ Risk Assessment")
        
        churn = predictions[0]
        inactive = predictions[6] if len(predictions) > 6 else 0
        complain = predictions[3] if len(predictions) > 3 else 0
        
        # Calculate risk score
        risk_score = 0
        if churn == 1:
            risk_score += 50
        if inactive == 1:
            risk_score += 25
        if complain == 1:
            risk_score += 25
        
        # Display risk meter
        if risk_score >= 50:
            st.error(f"🔴 **HIGH RISK** - Risk Score: {risk_score}% - Customer likely to churn! Immediate action required!")
        elif risk_score >= 25:
            st.warning(f"🟡 **MEDIUM RISK** - Risk Score: {risk_score}% - Customer needs attention")
        else:
            st.success(f"🟢 **LOW RISK** - Risk Score: {risk_score}% - Customer appears stable")
        
        # Business Recommendations
        st.markdown("---")
        st.subheader("💡 Business Recommendations")
        
        recommendations = []
        
        if churn == 1:
            recommendations.append("🚨 **URGENT:** Offer special discount and loyalty reward")
            recommendations.append("📞 Contact customer personally for feedback")
            recommendations.append("📡 Improve network service quality in their area")
        
        if predictions[1] == 1:  # Buy new plan
            recommendations.append("📱 Recommend suitable plan with upgrade discount")
        
        if predictions[3] == 1:  # Complain
            recommendations.append("📞 Proactively check network issues before they complain")
            recommendations.append("🎧 Ensure support team is ready to handle their call")
        
        if predictions[4] == 1:  # More data
            recommendations.append("📊 Recommend higher data plan or unlimited pack")
        
        if predictions[5] == 1:  # Upgrade
            recommendations.append("⬆️ Offer premium plan benefits with limited-time discount")
        
        if predictions[7] == 1:  # Respond to offer
            recommendations.append("🎯 Send personalized marketing campaigns")
        
        if predictions[8] == 1:  # High value
            recommendations.append("⭐ Provide VIP support and exclusive offers")
        
        if predictions[2] == 0:  # Will not recharge
            recommendations.append("💳 Send recharge reminder with cashback offer")
        
        if recommendations:
            for rec in recommendations[:6]:
                st.info(rec)
        else:
            st.success("✅ Customer appears satisfied. Maintain regular engagement and send periodic offers.")
        
        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        positive_count = sum(predictions[:10])
        
        with col1:
            st.metric("Positive Behaviors", f"{positive_count}/10")
        with col2:
            if churn == 1:
                st.metric("Churn Risk", "HIGH", delta="Action Needed")
            else:
                st.metric("Churn Risk", "LOW", delta="Stable")
        with col3:
            if predictions[1] == 1 or predictions[5] == 1:
                st.metric("Sales Opportunity", "YES", delta="Upsell")
            else:
                st.metric("Sales Opportunity", "NO", delta="Monitor")
        with col4:
            if predictions[8] == 1:
                st.metric("Customer Value", "HIGH", delta="VIP")
            else:
                st.metric("Customer Value", "Standard", delta="Regular")

else:
    st.info("👈 **Enter customer details in the sidebar and click 'Predict Customer Behavior'**")
    
    # Show example
    with st.expander("📖 How to use this app"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Enter customer information** in the left sidebar
        2. **Click the Predict button** at the bottom of the sidebar
        3. **View results** showing all 10 predictions
        4. **Read recommendations** for business actions
        
        ### Example Customer (Try this!):
        - Age: 35
        - Tenure: 24 months
        - Call Minutes: 800
        - Data Usage: 25 GB
        - Monthly Bill: ₹1200
        - Recharge Frequency: 4
        - Complaints: 0
        - Call Drops: 1
        - Rating: 5
        - App Logins: 15
        - Website Visits: 8
        - Offer Clicks: 5
        
        ### What the predictions mean:
        
        | Prediction | Business Action |
        |------------|----------------|
        | **Churn = Yes** | Immediate retention campaign needed |
        | **Buy New Plan = Yes** | Upsell opportunity |
        | **Recharge = No** | Send reminder with offer |
        | **Complain = Yes** | Proactive support outreach |
        | **Use More Data = Yes** | Recommend higher data plan |
        | **Upgrade = Yes** | Premium plan offer |
        | **Inactive = Yes** | Engagement campaign |
        | **Respond to Offer = Yes** | Marketing target |
        | **High Value = Yes** | VIP treatment |
        | **Contact Support = Yes** | Ensure support availability |
        """)
    
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Model Details:**
        - **Algorithm:** Random Forest Classifier (Multi-Output)
        - **Training Data:** 5,000+ synthetic customer records
        - **Features:** 60+ input features + 8 engineered features
        - **Output:** 10 customer behavior predictions simultaneously
        
        **Model Performance:**
        - Average Accuracy: ~85%
        - Average F1-Score: ~82%
        - Best performing on: Churn Prediction, Recharge Prediction
        """)

st.markdown("---")
st.caption("© 2024 Telecom Analytics | Customer Behavior Prediction System | Powered by Machine Learning")
