import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import joblib as jb
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, precision_score
from aqi_model import AQIDataLoader, AQIPredictor, AQIAnalyzer

def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

st.set_page_config(layout="wide")

image = Image.open(r"D:\ADYPU\SPCR training\19 Jan to 6 Feb\Project\Pune Image.jpeg")

# Display the image in the app
st.image(image, width="stretch")
# Load Data
loader = AQIDataLoader(
    r"D:\ADYPU\SPCR training\19 Jan to 6 Feb\Project\2024_hourly_data.csv",
    r"D:\ADYPU\SPCR training\19 Jan to 6 Feb\Project\aqi_data_pune_2017_to_2024.csv"
)

df = loader.load_and_prepare_data()

features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'NH3', 'OZONE']
X = df[features]
y = df['AQI']

# Load Model
predictor = AQIPredictor()
predictor.model = jb.load(r"D:\ADYPU\SPCR training\19 Jan to 6 Feb\Project\aqi_model.pkl")

y_pred = predictor.model.predict(X)
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

analyzer = AQIAnalyzer(df)

# HEADER
st.title("Pune AQI Analytics Dashboard")

# TOP INFORMATION
col1, col2 = st.columns(2)
y_cat = y.apply(categorize_aqi)
y_pred_cat = pd.Series(y_pred).apply(categorize_aqi)
accuracy = accuracy_score(y_cat, y_pred_cat)
precision = precision_score(
    y_cat, y_pred_cat,
    average='weighted'
)
with col1:
    st.subheader("R² Score:")
    st.write(f"{round(r2, 2)*100}%")
    st.subheader("RMSE:")
    st.write(f"{round(rmse, 2)}")

with col2:
    st.subheader("Accuracy:")
    st.write(f"{accuracy*100:.2f}%")
    st.subheader("Precision:")
    st.write(f"{precision*100:.2f}%")



# GRAPHS SECTION
st.subheader("Monthly AQI Trend")
monthly_data = analyzer.monthly_trend()
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(len(monthly_data)), monthly_data.values, color='#FFA500', marker='o', markerfacecolor='white', markeredgecolor='orange')
ax.set_xticks(range(len(months)))
ax.set_xticklabels(months)
ax.set_xlabel("Months", fontsize=20)
ax.set_ylabel("AQI")
ax.set_title("Monthly AQI Trend")
st.pyplot(fig)

st.subheader("Pollutant Correlation")
corr = analyzer.correlation_matrix()
fig, ax = plt.subplots(figsize=(12, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# PREDICTION SECTION
st.subheader("Predict AQI")

col1, col2, col3, col4 = st.columns(4)

with col1:
    pm25 = st.number_input("PM2.5", min_value=0.0)
    pm10 = st.number_input("PM10", min_value=0.0)

with col2:
    no2 = st.number_input("NO2", min_value=0.0)
    so2 = st.number_input("SO2", min_value=0.0)

with col3:
    co = st.number_input("CO", min_value=0.0)
    nh3 = st.number_input("NH3", min_value=0.0)

with col4:
    ozone = st.number_input("OZONE", min_value=0.0)

if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, so2, co, nh3, ozone]])
    prediction = predictor.predict(input_data)
    st.success(f"Predicted AQI: {round(prediction[0], 2)}")
    if prediction[0] <= 50:
        st.info("Air Quality: Good")
    elif 50 < prediction[0] <= 100:
        st.warning("Air Quality: Moderate")
    elif 100 < prediction[0] <= 150:
        st.error("Air Quality: Unhealthy for Sensitive Groups")
    elif 150 < prediction[0] <= 200:  
        st.error("Air Quality: Unhealthy")
    elif 200 < prediction[0] <= 300:
        st.error("Air Quality: Very Unhealthy")
    else:
        st.error("Air Quality: Hazardous")
