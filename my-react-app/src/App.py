from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pymysql
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import shap
from joblib import dump, load
from faker import Faker
import random
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

label_encoder = LabelEncoder()
scaler = StandardScaler()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# model = load('./Updated_random_forest_model.joblib')
# Set a random seed for reproducibility
np.random.seed(42)

# Initialize Faker for generating fake data
fake = Faker()

# Number of rows in the synthetic dataset
num_rows = 10000

# Generate synthetic data for each feature
data = {
    'Customer_Due_Diligence_CDD': np.random.choice([0, 1], size=num_rows),
    'AML_Compliance': np.random.choice([0, 1], size=num_rows),
    'KYC': np.random.choice([0, 1], size=num_rows),
    'Privacy_Policy_Compliance': np.random.choice([0, 1], size=num_rows),
    'Data_Encryption': np.random.choice([0, 1], size=num_rows),
    'Capital_Adequacy_Ratio_CAR': np.random.uniform(8, 15, size=num_rows),
    'Basel_III_Compliance': np.random.choice([0, 1], size=num_rows),
    'Reporting_to_Regulatory_Authorities': np.random.choice([0, 1], size=num_rows),
    'Liquidity_Risk_Management': np.random.choice([0, 1], size=num_rows),
    'Interest_Rate_Risk_Management': np.random.choice([0, 1], size=num_rows),
    'Fair_Lending_Practices': np.random.choice([0, 1], size=num_rows),
    'Risk_Based_Internal_Audits': np.random.choice([0, 1], size=num_rows),
    'Cybersecurity_Policies_and_Procedures': np.random.choice([0, 1], size=num_rows),
    'Insider_Trading_Prevention': np.random.choice([0, 1], size=num_rows),
    'Compliant_Sales_Practices': np.random.choice([0, 1], size=num_rows),
    'clear_accurate_info': np.random.choice([0, 1], size=num_rows),
    'effective_complaint_handling': np.random.choice([0, 1], size=num_rows),
    'suitability_financial_products': np.random.choice([0, 1], size=num_rows),
    'data_privacy_security': np.random.choice([0, 1], size=num_rows),
    'product_approval_adherence': np.random.choice([0, 1], size=num_rows),
    'customer_satisfaction_index': np.random.uniform(70, 100, size=num_rows),
    'complaint_resolution_time_days': np.random.uniform(1, 30, size=num_rows),
    'product_suitability_score': np.random.uniform(0, 10, size=num_rows),
    'data_security_expenditure_perc': np.random.uniform(5, 15, size=num_rows),
    'product_approval_rate_perc': np.random.uniform(80, 100, size=num_rows),
    'marketing_compliance': np.random.choice([0, 1], size=num_rows),
    'transparent_communication': np.random.choice([0, 1], size=num_rows),
    'advertisement_accuracy_score': np.random.uniform(0, 10, size=num_rows),
    'customer_communication_score': np.random.uniform(0, 10, size=num_rows),
    'stakeholder_engagement_score': np.random.uniform(0, 10, size=num_rows),
    'public_transparency_score': np.random.uniform(0, 10, size=num_rows),
    'social_media_compliance': np.random.choice([0, 1], size=num_rows),
    'regulatory_disclosure': np.random.choice([0, 1], size=num_rows),
    'TransactionAmount': np.random.uniform(1000, 100000, size=num_rows),
    'TransactionType': np.random.choice(['Deposit', 'Withdrawal'], size=num_rows),
    'IsSuspicious': np.random.choice([0, 1], size=num_rows),
    'EmployeeCount': np.random.randint(50, 500, size=num_rows),
    'CyberSecurityBudget': np.random.uniform(50000, 500000, size=num_rows),
    'IncidentSeverity': np.random.choice(['Low', 'Medium', 'High'], size=num_rows),
    'VulnerabilityCount': np.random.randint(0, 50, size=num_rows),
    'SolvencyRatio': np.random.uniform(0.1, 1.0, size=num_rows),
    'Audit_Committee_Existence': np.random.choice([0, 1], size=num_rows),
    'Internal_Audit_Function': np.random.choice([0, 1], size=num_rows),
    'Code_of_Ethics_Policy': np.random.choice([0, 1], size=num_rows),
    'Whistleblower_Policy': np.random.choice([0, 1], size=num_rows),
    'Risk_Management_Framework': np.random.choice([0, 1], size=num_rows),
    'Conflict_of_Interest_Disclosure': np.random.choice([0, 1], size=num_rows),
    'Related_Party_Transactions_Monitoring': np.random.choice([0, 1], size=num_rows),
    'Executive_Compensation_Disclosure': np.random.choice([0, 1], size=num_rows),
    'Shareholder_Rights_Protection': np.random.choice([0, 1], size=num_rows),
    'Governance_Policies_Disclosure': np.random.choice([0, 1], size=num_rows),
    'Succession_Planning': np.random.choice([0, 1], size=num_rows),
    'ComplianceStatus': np.random.choice(['Compliant', 'Non-Compliant'], size=num_rows),
}

# Create a DataFrame from the generated data
df = pd.DataFrame(data)

X = df.drop('ComplianceStatus', axis=1)
y = df['ComplianceStatus']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

categorical_columns = ['TransactionType', 'IncidentSeverity']
for column in categorical_columns:
    X[column] = label_encoder.fit_transform(X[column])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

print(classification_report(y_test, y_pred))

db = pymysql.connect(host='localhost',
                     user='aryapg',
                     password='arya440022',
                     db='compliance_status',
                     charset='utf8mb4',
                     cursorclass=pymysql.cursors.DictCursor)

# Create a table to store company data (customize based on your data structure)
with db.cursor() as cursor:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS compliance_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            Customer_Due_Diligence_CDD INT,
            AML_Compliance INT,
            KYC INT,
            Privacy_Policy_Compliance INT,
            Data_Encryption INT,
            Capital_Adequacy_Ratio_CAR FLOAT,
            Basel_III_Compliance INT,
            Reporting_to_Regulatory_Authorities INT,
            Liquidity_Risk_Management INT,
            Interest_Rate_Risk_Management INT,
            Fair_Lending_Practices INT,
            Risk_Based_Internal_Audits INT,
            Cybersecurity_Policies_and_Procedures INT,
            Insider_Trading_Prevention INT,
            Compliant_Sales_Practices INT,
            clear_accurate_info INT,
            effective_complaint_handling INT,
            suitability_financial_products INT,
            data_privacy_security INT,
            product_approval_adherence INT,
            customer_satisfaction_index FLOAT,
            complaint_resolution_time_days FLOAT,
            product_suitability_score FLOAT,
            data_security_expenditure_perc FLOAT,
            product_approval_rate_perc FLOAT,
            marketing_compliance INT,
            transparent_communication INT,
            advertisement_accuracy_score FLOAT,
            customer_communication_score FLOAT,
            stakeholder_engagement_score FLOAT,
            public_transparency_score FLOAT,
            social_media_compliance INT,
            regulatory_disclosure INT,
            TransactionAmount FLOAT,
            TransactionType VARCHAR(255),
            IsSuspicious INT,
            EmployeeCount INT,
            CyberSecurityBudget FLOAT,
            IncidentSeverity VARCHAR(255),
            VulnerabilityCount INT,
            SolvencyRatio FLOAT,
            Audit_Committee_Existence INT,
            Internal_Audit_Function INT,
            Code_of_Ethics_Policy INT,
            Whistleblower_Policy INT,
            Risk_Management_Framework INT,
            Conflict_of_Interest_Disclosure INT,
            Related_Party_Transactions_Monitoring INT,
            Executive_Compensation_Disclosure INT,
            Shareholder_Rights_Protection INT,
            Governance_Policies_Disclosure INT,
            Succession_Planning INT,
            ComplianceStatus VARCHAR(255),
            TopFeatures VARCHAR(255),
            SummaryText TEXT
        )
    ''')
    db.commit()

@app.route('/', methods=['POST'])
def upload_file():
    try:
        uploaded_file = request.files['file']
        df = pd.read_excel(uploaded_file)
        sample_data = df.drop('ComplianceStatus', axis=1)
        categorical_columns = ['TransactionType', 'IncidentSeverity']
        for column in categorical_columns:
            sample_data[column] = label_encoder.fit_transform(sample_data[column])
        sample_pred_proba = rf_model.predict(sample_data)
        sample_pred = (sample_pred_proba > 0.5).astype(int)
        custom_label_mapping = {0: 'Not compliant', 1: 'Compliant'}
        sample_pred_decoded = np.vectorize(custom_label_mapping.get)(sample_pred.flatten())
        with db.cursor() as cursor:
            for index, prediction in enumerate(sample_pred):
                compliance_status = 'Compliant' if prediction >= 0.5 else 'Not Compliant'
                get_last_id_query = "SELECT id FROM compliance_data ORDER BY id DESC LIMIT 1"
                cursor.execute(get_last_id_query)
                last_inserted_id = cursor.fetchone()['id']
                update_query = "UPDATE compliance_data SET ComplianceStatus = %s WHERE id = %s"
                cursor.execute(update_query, (compliance_status, last_inserted_id))
        db.commit()
        explainer = shap.Explainer(rf_model)
        shap_values = explainer.shap_values(sample_data.iloc[0])
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_names = sample_data.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
        feature_importance_df_sorted = feature_importance_df.sort_values(by='Importance', ascending=False)
        top_features = feature_importance_df_sorted['Feature'].head(3).tolist()
        summary_text = "Based on our analysis, the major factors affecting compliance status are:\n"
        for i, feature in enumerate(top_features):
            if feature_importance_df_sorted.loc[feature_importance_df_sorted['Feature'] == feature, 'Importance'].values[0] > 0:
                effect = "Improving"
            else:
                effect = "decreasing"
            summary_text += f"{i+1}. {feature}: {effect} this feature would contribute to compliance.\n"
        predicted_labels = np.vectorize(custom_label_mapping.get)(sample_pred)
        predicted_label = predicted_labels[0]
        # Extract top features and summary text
        top_features_str = ', '.join(top_features)
        summary_text = "Based on our analysis, the major factors affecting compliance status are:\n"
        for i, feature in enumerate(top_features):
            if feature_importance_df_sorted.loc[feature_importance_df_sorted['Feature'] == feature, 'Importance'].values[0] > 0:
                effect = "Improving"
            else:
                effect = "decreasing"
            summary_text += f"{i+1}. {feature}: {effect} this feature would contribute to compliance.\n"

        # Insert data into the database
        with db.cursor() as cursor:
            placeholders = ', '.join(['%s'] * len(df.columns)) + ', %s, %s'  # Update placeholders
            columns = ', '.join(df.columns) + ', TopFeatures, SummaryText'  # Update columns
            query = f"INSERT INTO compliance_data ({columns}) VALUES ({placeholders})"
            values = [tuple(x) + (top_features_str, summary_text) for x in df.to_numpy()]  # Update values
            cursor.executemany(query, values)
            db.commit()
        return jsonify({
            "success": True,
            "complianceStatus": predicted_label,
            "topFeatures": top_features,
            "summaryText": summary_text
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
