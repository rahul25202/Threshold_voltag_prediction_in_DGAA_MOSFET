import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("dgaa_mosfet_threshold_voltage_dataset.csv")

# Prepare the data - ensure we're only using the 6 input features
X = df[['Channel_Length_nm', 'Channel_Thickness_nm', 'Oxide_Thickness_nm', 
        'Inner_Gate_Radius_nm', 'Channel_Doping_cm3', 'Drain_Voltage_V']]
y = df["Threshold_Voltage_V"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Initialize and train the model
model = RandomForestRegressor(
    n_estimators=2000,
    random_state=42,
    max_leaf_nodes=1000,
    max_depth=100
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

def predict_threshold_voltage():
    print("\nEnter MOSFET parameters to predict threshold voltage:")
    
    channel_length = float(input("Channel Length (nm): "))
    channel_thickness = float(input("Channel Thickness (nm): "))
    oxide_thickness = float(input("Oxide Thickness (nm): "))
    inner_gate_radius = float(input("Inner Gate Radius (nm): "))
    channel_doping = float(input("Channel Doping (cm^-3): "))
    drain_voltage = float(input("Drain Voltage (V): "))
    
    input_data = pd.DataFrame([[channel_length, channel_thickness, oxide_thickness, 
                              inner_gate_radius, channel_doping, drain_voltage]],
                            columns=X_train.columns)
    
    threshold_voltage = model.predict(input_data)
    print(f"\nPredicted Threshold Voltage: {threshold_voltage[0]:.4f} V")

def plot_parameter_effects():
    median_values = X.median()
    
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 18))
    fig.suptitle('Threshold Voltage vs MOSFET Parameters', y=1.02, fontsize=16)
    
    def plot_parameter(ax, param_name, title, xlabel, log_scale=False):
        param_range = np.linspace(X[param_name].min(), X[param_name].max(), 100)
        if log_scale:
            param_range = np.logspace(np.log10(X[param_name].min()), 
                                    np.log10(X[param_name].max()), 100)
        
        # Create DataFrame with correct column names
        input_data = pd.DataFrame(np.tile(median_values.values, (100, 1)), 
                                columns=X.columns)
        input_data[param_name] = param_range
        
        predictions = model.predict(input_data)
        
        ax.plot(param_range, predictions, color='royalblue', linewidth=2)
        ax.set_title(title, pad=20)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Threshold Voltage (V)')
        ax.grid(True, alpha=0.3)
        
        if log_scale:
            ax.set_xscale('log')
        
        ax.axvline(median_values[param_name], color='red', linestyle='--', alpha=0.7)
        ax.text(median_values[param_name], ax.get_ylim()[0], ' Median', 
                color='red', va='bottom', ha='left')
    
    plot_parameter(axes[0, 0], 'Channel_Length_nm', 
                  'Threshold Voltage vs Channel Length', 'Channel Length (nm)')
    plot_parameter(axes[0, 1], 'Channel_Thickness_nm', 
                  'Threshold Voltage vs Channel Thickness', 'Channel Thickness (nm)')
    plot_parameter(axes[1, 0], 'Oxide_Thickness_nm', 
                  'Threshold Voltage vs Oxide Thickness', 'Oxide Thickness (nm)')
    plot_parameter(axes[1, 1], 'Inner_Gate_Radius_nm', 
                  'Threshold Voltage vs Inner Gate Radius', 'Inner Gate Radius (nm)')
    plot_parameter(axes[2, 0], 'Channel_Doping_cm3', 
                  'Threshold Voltage vs Channel Doping', 'Channel Doping (cm$^{-3}$)', 
                  log_scale=True)
    plot_parameter(axes[2, 1], 'Drain_Voltage_V', 
                  'Threshold Voltage vs Drain Voltage', 'Drain Voltage (V)')
    
    plt.tight_layout()
    plt.show()

while True:
    print("\nOptions:")
    print("1. Predict threshold voltage")
    print("2. View parameter effect plots")
    print("3. Exit")
    
    choice = input("Enter your choice (1, 2, or 3): ")
    
    if choice == '1':
        predict_threshold_voltage()
    elif choice == '2':
        plot_parameter_effects()
    elif choice == '3':
        print("Exiting program...")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")