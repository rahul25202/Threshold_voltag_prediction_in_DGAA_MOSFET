from sklearn.ensemble import RandomForestRegressor as Rf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE, r2_score
import numpy as np

# Load data
df = pd.read_csv("dgaa_mosfet_threshold_voltage_dataset.csv")

# Prepare data
X = df.drop("Threshold_Voltage_V", axis=1)
y = df["Threshold_Voltage_V"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Train model
model = Rf(n_estimators=2000, random_state=42, max_leaf_nodes=1000, max_depth=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(MSE(y_test, y_pred))

print("\nModel Evaluation Metrics:")
print("R² Score:", r2)
print("RMSE:", rmse)

# Function to get user input and make prediction
def predict_threshold_voltage():
    print("\nEnter MOSFET parameters to predict threshold voltage:")
    
    # Get user input for each feature
    channel_length = float(input("Channel Length (nm): "))
    channel_thickness = float(input("Channel Thickness (nm): "))
    oxide_thickness = float(input("Oxide Thickness (nm): "))
    inner_gate_radius = float(input("Inner Gate Radius (nm): "))
    channel_doping = float(input("Channel Doping (cm^-3): "))
    drain_voltage = float(input("Drain Voltage (V): "))
    
    # Create input array
    input_data = np.array([[
        channel_length, 
        channel_thickness, 
        oxide_thickness, 
        inner_gate_radius, 
        channel_doping, 
        drain_voltage
    ]])
    
    # Make prediction
    threshold_voltage = model.predict(input_data)
    
    print(f"\nPredicted Threshold Voltage: {threshold_voltage[0]:.4f} V")

# Main execution loop
while True:
    print("\nOptions:")
    print("1. Predict threshold voltage")
    print("2. Exit")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        predict_threshold_voltage()
    elif choice == '2':
        print("Exiting program...")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")