import pandas as pd
import numpy as np
import joblib as jb

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

class AQIDataLoader:
    def __init__(self, hourly_file, daily_file):
        self.hourly_file = hourly_file
        self.daily_file = daily_file

    def load_and_prepare_data(self):
        hourly_df = pd.read_csv(self.hourly_file, on_bad_lines='skip')
        hourly_df['Date'] = pd.to_datetime(hourly_df['Date'], errors='coerce')

        # Drop rows with invalid dates
        hourly_df = hourly_df.dropna(subset=['Date'])

        # Drop the 'Time' column as it's not needed for averaging
        hourly_df = hourly_df.drop(columns=['Time'])

        # Convert pollutant columns to numeric, coercing errors to NaN
        pollutant_cols = ['CO', 'NH3', 'NO2', 'OZONE', 'PM10', 'PM2.5', 'SO2']
        for col in pollutant_cols:
            hourly_df[col] = pd.to_numeric(hourly_df[col], errors='coerce')

        # Convert hourly → daily average
        daily_pollutants = hourly_df.groupby('Date').mean().reset_index()

        # Load daily AQI dataset
        aqi_df = pd.read_csv(self.daily_file, on_bad_lines='skip')
        aqi_df['Date'] = pd.to_datetime(aqi_df['Date'])

        # Merge pollutant data with AQI
        merged_df = pd.merge(
            daily_pollutants,
            aqi_df[['Date', 'AQI']],
            on='Date',
            how='inner'
        )

        # Remove missing AQI values
        merged_df = merged_df.dropna()

        return merged_df

# OOP PART 2 : Analyzer Class
class AQIAnalyzer:

    def __init__(self, df):
        self.df = df

    def monthly_trend(self):
        self.df['Month'] = self.df['Date'].dt.month
        return self.df.groupby('Month')['AQI'].mean()

    def correlation_matrix(self):
        return self.df.corr(numeric_only=True)

# OOP PART 3 : ML Model Class
class AQIPredictor:

    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=150, random_state=42)

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return r2, rmse

    def predict(self, data):
        return self.model.predict(data)

    def save_model(self):
        jb.dump(self.model, "aqi_model.pkl")

    


if __name__ == "__main__":
    # Load Data Using Class
    loader = AQIDataLoader(
        "2024_hourly_data.csv",
        "aqi_data_pune_2017_to_2024.csv"
    )
    df = loader.load_and_prepare_data()
    
    # Features & Target
    features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'NH3', 'OZONE']
    X = df[features]
    y = df['AQI']

    # Train Model
    predictor = AQIPredictor()
    r2, rmse = predictor.train(X, y)
    predictor.save_model()

    print(f"Model trained with R2: {r2:.3f}, RMSE: {rmse:.3f}")
