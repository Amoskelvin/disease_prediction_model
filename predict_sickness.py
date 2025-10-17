import pandas as pd
import joblib

model = joblib.load('sickness_predict.joblib')
data = pd.read_csv('medical_records.csv')

symptoms = [col for col in data.columns if col not in ['label']]

def check_sickness():
    print("\n=== PATIENT HEALTH DATA ENTRY ===")
    print("Please answer carefully. Type 'yes' or 'no' for symptoms.\n")

    user_data = {}

    for symptom in symptoms:
        # AGE
        if symptom.lower() == 'age':
            while True:
                age = input('Patient age > ').strip()
                if age.isdigit() and 0 < int(age) < 120:
                    user_data[symptom] = int(age)
                    break
                else:
                    print("⚠️  Enter a valid age between 1 and 120.")

        # SEX
        elif symptom.lower() == 'sex':
            while True:
                sex = input('Sex (M/F) > ').strip().upper()
                if sex in ['M', 'F']:
                    user_data[symptom] = 0 if sex == 'M' else 1
                    break
                else:
                    print("⚠️  Invalid input. Please type 'M' or 'F'.")

        # SYMPTOMS (YES/NO)
        else:
            while True:
                val = input(f'{symptom} (Yes/No) > ').strip().lower()
                if val in ['yes', 'no']:
                    user_data[symptom] = 1 if val == 'yes' else 0
                    break
                else:
                    print("⚠️  Please answer with 'yes' or 'no' only.")

    # Convert to DataFrame and predict
    patient_symptoms = pd.DataFrame([user_data])
    prediction = model.predict(patient_symptoms)[0]

    print("\n✅ Prediction complete!")
    print(f"🩺 The model predicts: {prediction}")

check_sickness()
