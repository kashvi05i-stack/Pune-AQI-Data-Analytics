# Pune AQI Analysis & Prediction System

![image alt](https://github.com/kashvi05i-stack/Pune-AQI-Data-Analytics/blob/00391604095d90e3cd35292896f1ef8fba90b532/Header.png)

![image alt](https://github.com/kashvi05i-stack/Pune-AQI-Data-Analytics/blob/00391604095d90e3cd35292896f1ef8fba90b532/Metrics_2.png)

![image alt](https://github.com/kashvi05i-stack/Pune-AQI-Data-Analytics/blob/00391604095d90e3cd35292896f1ef8fba90b532/Monthly%20Trend_3.png)

![image alt](https://github.com/kashvi05i-stack/Pune-AQI-Data-Analytics/blob/00391604095d90e3cd35292896f1ef8fba90b532/Pollutant%20Corelation_4.png)

![image alt](https://github.com/kashvi05i-stack/Pune-AQI-Data-Analytics/blob/00391604095d90e3cd35292896f1ef8fba90b532/Prediction_5.png)

## Overview
This project is an Air Quality Index (AQI) Analysis and Prediction System developed using Python, Machine Learning, and Streamlit. The project analyzes Pune city air quality data, performs pollutant trend analysis, visualizes AQI patterns, and predicts AQI levels using a Random Forest Regression model.

The system integrates data preprocessing, exploratory analysis, machine learning, and an interactive Streamlit dashboard for real-time AQI prediction and visualization.

---

## Features
- AQI data preprocessing and cleaning
- Hourly to daily pollutant aggregation
- Correlation analysis of pollutants
- Monthly AQI trend analysis
- Machine Learning-based AQI prediction
- Interactive Streamlit dashboard
- Real-time AQI prediction from pollutant inputs
- AQI category classification
- Heatmap and trend visualizations

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

---

## Dataset Used

### 1. 2024_hourly_data.csv
Contains hourly pollutant concentration data including:
- PM2.5
- PM10
- NO2
- SO2
- CO
- NH3
- OZONE

### 2. aqi_data_pune_2017_to_2024.csv
Contains:
- Date-wise AQI values
- Historical AQI records

---

## Project Structure
```text
Pune-AQI-Analysis
│
├── aqi_model.py
├── project_streamlit.py
├── 2024_hourly_data.csv
├── aqi_data_pune_2017_to_2024.csv
├── aqi_model.pkl
└── README.md
```

---

## Concepts Implemented
- Data Cleaning
- Feature Engineering
- Machine Learning Regression
- Random Forest Regressor
- Data Visualization
- Streamlit Dashboard Development
- Correlation Analysis
- Object-Oriented Programming (OOP)

---

## Functionalities

### 1. Data Loading & Cleaning
- Handles missing values
- Converts pollutant columns to numeric
- Aggregates hourly pollutant data into daily averages

### 2. AQI Analysis
- Monthly AQI trend analysis
- Pollutant correlation matrix generation
- AQI categorization

### 3. Machine Learning Model
- Random Forest Regression Model
- Train-test split
- Model evaluation using:
  - R² Score
  - RMSE

### 4. Interactive Dashboard
The Streamlit application provides:
- AQI analytics dashboard
- Pollutant trend graphs
- Correlation heatmap
- Real-time AQI prediction system

### 5. AQI Prediction
Users can input pollutant values:
- PM2.5
- PM10
- NO2
- SO2
- CO
- NH3
- OZONE

The model predicts:
- AQI value
- AQI category

---

## AQI Categories
| AQI Range | Category                       |
|-----------|--------------------------------|
| 0–50      | Good                           |
| 51–100    | Moderate                       |
| 101–150   | Unhealthy for Sensitive Groups |
| 151–200   | Unhealthy                      |
| 201–300   | Very Unhealthy                 |
| 300+      | Hazardous                      |

---

## Visualizations Included
- Monthly AQI Trend Graph
- Pollutant Correlation Heatmap
- AQI Prediction Results

---

## Machine Learning Metrics
The project evaluates model performance using:
- R² Score
- RMSE
- Accuracy
- Precision

---

## How to Run

### Install Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib pillow
```

---

## Step 1: Train the Model
Run:
```bash
python aqi_model.py
```

This generates:
```text
aqi_model.pkl
```

---

## Step 2: Run Streamlit Dashboard
```bash
streamlit run project_streamlit.py
```

---

## Sample Features
- Interactive dashboard interface
- Real-time AQI prediction
- Dynamic pollutant analysis
- Visual analytics

---

## Learning Outcomes
- Learned implementation of machine learning regression models
- Improved understanding of environmental data analysis
- Gained experience with Streamlit dashboard development
- Understood AQI prediction workflows
- Implemented data visualization and analytics techniques

---

## Future Improvements
- Real-time AQI API integration
- Deployment on cloud platforms
- Deep learning AQI prediction models
- Mobile-responsive dashboard
- Multi-city AQI analysis

---

## Author
Kashvi Chaturvedi
