import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌾 Soybean Disease PDI Forecasting App")

# Upload CSV
uploaded_file = st.file_uploader("Upload 'Disease_data_corrected4.csv'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocessing
    df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col or df[col].isnull().all()])
    varieties = ['JS-335', 'PK -472', 'Shivalik', 'JS93-05', 'JS 95-60', 'Punjab1', 'Monetta', 'NRC-7']
    for variety in varieties:
        if variety in df.columns:
            df[variety] = pd.to_numeric(df[variety], errors='coerce')
    df.fillna(method='ffill', inplace=True)

    # Encode Location
    le = LabelEncoder()
    df['Location'] = le.fit_transform(df['Location'])
    location_dict = dict(zip(le.classes_, le.transform(le.classes_)))

    # UI Inputs
    st.sidebar.header("User Input Parameters")
    selected_variety = st.sidebar.selectbox("Select Variety", varieties)
    selected_location_name = st.sidebar.selectbox("Select Location", sorted(location_dict.keys()))
    sowing_date = st.sidebar.date_input("Enter Sowing Date", value=datetime(2024, 7, 1).date())

    # Convert today to date object to fix datetime.date subtraction issue
    today = datetime.today().date()
    crop_days = (today - sowing_date).days
    crop_week = crop_days // 7
    current_smw = int(today.strftime("%U"))  # Standard Meteorological Week

    st.sidebar.markdown(f"**Crop Week:** {crop_week}")
    st.sidebar.markdown(f"**SMW:** {current_smw}")

    # Filter and define model data
    encoded_location = location_dict[selected_location_name]
    df = df[df[selected_variety].notnull()]  # Remove rows without target PDI

    # Define features and target
    features = ['Year', 'SMW', 'Crop_Week', 'Location', 'Longitude', 'Latitude',
                'Max_Temp', 'Min_Temp', 'Max_Humidity', 'Min_Humidity',
                'No_of_Rainy_Days', 'Rainfall', 'Wind_Velocity']
    X = df[features]
    y = df[selected_variety]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("📊 Model Evaluation")
    st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # Prepare user input for prediction
    latest_row = df[df['Location'] == encoded_location].sort_values(by='Year', ascending=False).iloc[0]
    pred_input = pd.DataFrame({
        'Year': [latest_row['Year']],
        'SMW': [current_smw],
        'Crop_Week': [crop_week],
        'Location': [encoded_location],
        'Longitude': [latest_row['Longitude']],
        'Latitude': [latest_row['Latitude']],
        'Max_Temp': [latest_row['Max_Temp']],
        'Min_Temp': [latest_row['Min_Temp']],
        'Max_Humidity': [latest_row['Max_Humidity']],
        'Min_Humidity': [latest_row['Min_Humidity']],
        'No_of_Rainy_Days': [latest_row['No_of_Rainy_Days']],
        'Rainfall': [latest_row['Rainfall']],
        'Wind_Velocity': [latest_row['Wind_Velocity']]
    })

    user_pred = model.predict(pred_input)[0]

    # Display prediction
    st.subheader("🧪 Prediction for Your Input")
    st.write(f"**Predicted PDI for `{selected_variety}` at `{selected_location_name}`:** `{user_pred:.2f}`")

    # Severity level
    if user_pred < 10:
        severity = "Low"
    elif 10 <= user_pred < 30:
        severity = "Medium"
    else:
        severity = "High"
    st.markdown(f"### 🌡️ Disease Severity: **{severity}**")

    # Actual vs Predicted Plot
    st.subheader("📉 Actual vs Predicted PDI")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, color='green')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual PDI")
    ax.set_ylabel("Predicted PDI")
    ax.set_title(f"Actual vs Predicted PDI - {selected_variety}")
    ax.grid(True)
    st.pyplot(fig)

else:
    st.info("📂 Please upload the 'Disease_data_corrected4.csv' file to continue.")
