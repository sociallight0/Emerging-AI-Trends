# iot_simulation.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def simulate_iot_system(output_path):
    """Simulate smart agriculture system with AI-IoT integration."""
    # Simulated sensor data
    np.random.seed(42)
    data = {
        "soil_moisture": np.random.uniform(20, 80, 1000),
        "temperature": np.random.uniform(15, 35, 1000),
        "humidity": np.random.uniform(30, 90, 1000),
        "crop_yield": np.random.uniform(800, 1500, 1000)  # kg/ha
    }
    df = pd.DataFrame(data)
    
    # Train model
    X = df[["soil_moisture", "temperature", "humidity"]]
    y = df["crop_yield"]
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict and evaluate
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Sample Predicted Crop Yield: {y_pred[0]:.2f} kg/ha")
    
    # Save data for reference
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    simulate_iot_system("data/iot_sensor_data.csv")
