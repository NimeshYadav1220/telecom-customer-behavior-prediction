# Telecom Customer Behavior Prediction System

## 📱 Project Overview

This machine learning project predicts various customer behaviors for a telecommunications company. The system helps the company understand customer intentions and take proactive actions to improve retention, increase revenue, and enhance customer satisfaction.

## 🎯 Objective

Build a machine learning system that analyzes telecom customer data and predicts future customer behavior including:
- Customer churn
- Plan purchases
- Recharge behavior
- Complaint likelihood
- Data usage patterns
- Plan upgrades/downgrades
- Customer inactivity
- Offer responses
- High-value customer identification
- Support contact likelihood

## 📊 Dataset

The system uses a synthetic telecom customer dataset with **5000+ records** containing:

### Features (60+ columns):
- **Customer Profile:** Age, gender, location, tenure, income, device type
- **Service Usage:** Call minutes, data usage, SMS count, roaming
- **Billing:** Monthly bill, recharge amount/frequency, payment method
- **Plan Details:** Current plan, data/call limits, add-ons
- **Complaints:** Count, type, resolution time, ratings
- **Network Quality:** Signal strength, call drops, internet speed
- **Engagement:** App logins, offer clicks, campaign responses
- **Risk Indicators:** Inactive days, port requests, usage drops

## 🛠️ Technologies Used

| Category | Technologies |
|----------|--------------|
| Language | Python 3.8+ |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Web App | Streamlit |
| Model Persistence | Joblib |

## 🤖 Algorithms Used

The project trains three algorithms using **MultiOutputClassifier**:
1. **Logistic Regression** - Baseline model
2. **Decision Tree** - Interpretable model
3. **Random Forest** - Ensemble model (Best performer)

### Model Performance
- Best Model: Random Forest
- Average Accuracy: ~85%
- Average F1-Score: ~82%

## 📁 Project Structure
