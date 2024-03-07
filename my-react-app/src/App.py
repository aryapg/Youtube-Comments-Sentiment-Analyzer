from flask import Flask, request, jsonify
from io import BytesIO
import pandas as pd
import numpy as np
import pymysql
from flask_cors import CORS
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import shap
import joblib

label_encoder = LabelEncoder()
scaler = StandardScaler()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the deep learning model
model = joblib.load('/home/arya/my-react-app/random_forest_model_updated.joblib')
# Configure MySQL
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
            ComplianceStatus VARCHAR(255)
        )
    ''')
    db.commit()

@app.route('/predict', methods=['POST'])
def upload_file():
    try:
        uploaded_file = request.files['file']
        df = pd.read_excel(uploaded_file)
        sample_data = df.drop('ComplianceStatus', axis=1)

        with db.cursor() as cursor:
            placeholders = ', '.join(['%s'] * len(df.columns))
            columns = ', '.join(df.columns)
            query = f"INSERT INTO compliance_data ({columns}) VALUES ({placeholders})"
            
            values = [tuple(x) for x in df.to_numpy()]
            cursor.executemany(query, values)
            db.commit()
            
        sample_pred_proba = model.predict_proba(sample_data)[:, 1]
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
    
        explainer = shap.Explainer(model)
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
        
        return render_template('result.html', prediction=predicted_label, top_features=top_features, summary_text=summary_text)
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)