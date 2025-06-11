import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import zipfile
import io

# App-Titel
st.title("aPTT-Vorhersage-App mit Zeitdifferenz-Simulation")

# --- Hochladen der Modell-ZIP-Datei ---
st.sidebar.header("1. Lade deine Modell-ZIP-Datei hoch")
uploaded_zip = st.sidebar.file_uploader("ZIP-Datei mit trainierten Modellen (joblib-Format)", type="zip")

if uploaded_zip:
    with zipfile.ZipFile(uploaded_zip, "r") as zip_ref:
        zip_ref.extractall("model_temp")
        try:
            xgb_model_1 = joblib.load("model_temp/xgb_model_1.pkl")
            xgb_model_2 = joblib.load("model_temp/xgb_model_2.pkl")
            xgb_classifier = joblib.load("model_temp/xgb_classifier.pkl")
        except Exception as e:
            st.error(f"Fehler beim Laden der Modelle: {e}")
            st.stop()
else:
    st.warning("Bitte lade die ZIP-Datei mit den Modellen hoch.")
    st.stop()

# --- Eingabe der Patientendaten ---
st.sidebar.header("2. Gib Patientendaten ein")

previous_aptt = st.sidebar.number_input("Aktueller aPTT-Wert", min_value=0.0, value=50.0)
previous_heparin = st.sidebar.number_input("Vorherige Heparin-Perfusor-Laufrate", min_value=0.0, value=1000.0)
current_heparin = st.sidebar.number_input("Aktuelle Heparin-Perfusor-Laufrate", min_value=0.0, value=2000.0)

# --- Hauptfunktion zur Vorhersage ---
def predict_for_patient(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df['Heparin_Value_diff'] = input_df['Heparin_Value'] - input_df['Previous_Heparin_Value']

    input_df['xgboost_pred'] = xgb_model_2.predict(input_df)
    input_df['elasticnet_pred'] = xgb_model_1.predict(input_df.drop(columns='xgboost_pred'))

    model_choice = xgb_classifier.predict(input_df.drop(columns='Time_Heparin_To_aPTT'))[0]

    input_df = input_df.drop(columns=['xgboost_pred', 'elasticnet_pred'])

    if model_choice == 0:
        model_name = "model_1"
        prediction = xgb_model_1.predict(input_df)[0]
    else:
        model_name = "model_2"
        prediction = xgb_model_2.predict(input_df)[0]

    return {
        "Modell": model_name,
        "Vorhergesagter Wert": int(prediction + input_df['Previous_aPTT_Value'].iloc[0]),
    }

# --- Simulation ---
def simulate_time_sweep(base_patient_data, time_range=np.arange(0, 24, 1)):
    predictions = []
    times = []

    for t in time_range:
        patient = base_patient_data.copy()
        patient['Time_Heparin_To_aPTT'] = t
        pred = predict_for_patient(patient)["Vorhergesagter Wert"]
        predictions.append(pred)
        times.append(t)

    lower_bound = np.array(predictions) - 5
    upper_bound = np.array(predictions) + 5

    return times, predictions, lower_bound, upper_bound

# --- Plot erzeugen ---
def plot_prediction(times, preds, lower, upper):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, preds, label='Vorhergesagter aPTT-Wert', color='blue')
    ax.fill_between(times, lower, upper, color='blue', alpha=0.2, label='±5 aPTT Fehlerintervall')
    ax.set_xlabel('Time_Heparin_To_aPTT (Stunden)')
    ax.set_ylabel('Vorhergesagter aPTT-Wert')
    ax.set_title('aPTT-Vorhersagen in den nächsten 24 Stunden')
    ax.legend()
    ax.grid(True)
    return fig

# --- Starte Simulation ---
st.header("aPTT-Vorhersage simulieren")

if st.button("Simulation starten"):
    patient_data = {
        'Previous_aPTT_Value': previous_aptt,
        'Previous_Heparin_Value': previous_heparin,
        'Heparin_Value': current_heparin,
    }
    with st.spinner("Simulation läuft..."):
        times, preds, lower, upper = simulate_time_sweep(patient_data)
        fig = plot_prediction(times, preds, lower, upper)
        st.pyplot(fig)



