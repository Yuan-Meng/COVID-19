import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pickle
import joblib

import warnings

warnings.simplefilter("ignore")

# + Collapsed="false" execution_event_id="38abdf1d-e409-4759-a947-9c5f67d72e60" last_executed_text="# Read data and models from pickles\nwith open(\"covid19_risk_xgboost.pkl\", \"rb\") as file:\n    xgb_best = pickle.load(file)\n\nwith open(\"age_scaler.pkl\", \"rb\") as file:\n    age_scaler = pickle.load(file)\n\nwith open(\"docs_scaler.pkl\", \"rb\") as file:\n    docs_scaler = pickle.load(file)\n\nwith open(\"beds_scaler.pkl\", \"rb\") as file:\n    beds_scaler = pickle.load(file)\n\nwith open(\"medical.pkl\", \"rb\") as file:\n    medical = pickle.load(file)" persistent_id="e858e76e-09e5-4b24-8721-703f39e9fb7e"
# Read data and models from pickles
with open("covid19_risk_xgboost.pkl", "rb") as file:
    xgb_best = pickle.load(file)

with open("age_scaler.pkl", "rb") as file:
    age_scaler = pickle.load(file)

with open("docs_scaler.pkl", "rb") as file:
    docs_scaler = pickle.load(file)

with open("beds_scaler.pkl", "rb") as file:
    beds_scaler = pickle.load(file)

with open("medical.pkl", "rb") as file:
    medical = pickle.load(file)


# + [markdown] Collapsed="false" persistent_id="ab07431f-492d-445b-9cf3-7dca8cbf760f"
# ## Take new input

# + Collapsed="false" execution_event_id="0feb82ae-f82d-47fb-a025-68c8dae36c18" last_executed_text="# Function: Take patient info\ndef patient_info():\n\n    # Demographic info\n    age = input(\"What's your age?\")\n    sex = input(\"What's your biological sex? (female: 1, male: 0)\")\n    country = input(\"Which country do you live in?\")\n    docs_per_10k = medical.loc[\n        medical[\"country\"].str.contains(country), \"docs_per_10k\"\n    ].item()\n    beds_per_10k = medical.loc[\n        medical[\"country\"].str.contains(country), \"beds_per_10k\"\n    ].item()\n\n    # Pre-conditions\n    preconition = input(\"Any chronic dieases that you know of? (yes: 1, no: 0)\")\n    hypertension = input(\"Do you have hypertension? (yes: 1, no: 0)\")\n    diabetes = input(\"Do you have diabetes? (yes: 1, no: 0)\")\n    heart = input(\"Do you have heart diseases? (yes: 1, no: 0)\")\n\n    # Symptoms\n    fever = input(\"Do you have fever? (yes: 1, no: 0)\")\n    cough = input(\"Are you coughing? (yes: 1, no: 0)\")\n    fatigue = input(\"Do you feel fatigue? (yes: 1, no: 0)\")\n    sore_throat = input(\"Do you have a sore throat? (yes: 1 , no:0)\")\n\n    # Save results in a list\n    return {\n        \"sex\": int(sex),\n        \"chronic_disease_binary\": int(preconition),\n        \"beds_per_10k\": float(beds_scaler.transform([[beds_per_10k]])),\n        \"docs_per_10k\": float(docs_scaler.transform([[docs_per_10k]])),\n        \"hypertension\": int(hypertension),\n        \"diabetes\": int(diabetes),\n        \"heart\": int(heart),\n        \"fever\": int(fever),\n        \"cough\": int(cough),\n        \"fatigue\": int(fatigue),\n        \"sore throat\": int(sore_throat),\n        \"age_scaled\": float(age_scaler.transform([[age]])),\n    }\n\n\n# Function: Output risk\ndef risk_predictor():\n\n    # Get patient info\n    new_case = patient_info()\n\n    # Convert to useable format\n    new_case = pd.DataFrame(new_case, index=[0])\n\n    # Make prediction\n    new_prediction = xgb_best.predict_proba(new_case)\n\n    # Print result\n    print(\n        \"Once contracted COVID-19, your mortality risk is {}%.\".format(\n            round(new_prediction.tolist()[0][1] * 100, 2)\n        )\n    )" persistent_id="456dfe32-6fc2-42c6-b60f-29ad83b7be37"
# Function: Take patient info
def patient_info():

    # Demographic info
    age = input("What's your age?")
    sex = input("What's your biological sex? (female: 1, male: 0)")

    # Pre-conditions
    preconition = input("Any chronic dieases that you know of? (yes: 1, no: 0)")
    hypertension = input("Do you have hypertension? (yes: 1, no: 0)")
    diabetes = input("Do you have diabetes? (yes: 1, no: 0)")
    heart = input("Do you have heart diseases? (yes: 1, no: 0)")

    # Symptoms
    fever = input("Do you have fever? (yes: 1, no: 0)")
    cough = input("Are you coughing? (yes: 1, no: 0)")
    fatigue = input("Do you feel fatigue? (yes: 1, no: 0)")
    sore_throat = input("Do you have a sore throat? (yes: 1 , no:0)")

    # Save results in a list
    return {
        "sex": int(sex),
        "chronic_disease_binary": int(preconition),
        "hypertension": int(hypertension),
        "diabetes": int(diabetes),
        "heart": int(heart),
        "fever": int(fever),
        "cough": int(cough),
        "fatigue": int(fatigue),
        "sore throat": int(sore_throat),
        "age_scaled": float(age_scaler.transform([[age]])),
    }


# Function: Output risk
def risk_predictor():

    # Get patient info
    new_case = patient_info()

    # Convert to useable format
    new_case = pd.DataFrame(new_case, index=[0])

    # Make prediction
    new_prediction = xgb_best.predict_proba(new_case)

    # Print result
    print(
        "Once contracted COVID-19, your mortality risk is {}%.".format(
            round(new_prediction.tolist()[0][1] * 100, 2)
        )
    )

# + [markdown] Collapsed="false" persistent_id="76d3b7ab-78f3-4bda-9e55-06fadea80a56"
# ## Try it for yourself

# + Collapsed="false" execution_event_id="4dfeb895-69bc-4244-91d2-4c491d00ae09" last_executed_text="risk_predictor()" persistent_id="01ccd8ad-3ba5-415e-a100-a0e3639e8605"
# Uncomment the function below 
# risk_predictor()

# + Collapsed="false" persistent_id="fc1246bf-5b79-4ca6-8168-f329b1cc82d9"

