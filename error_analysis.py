import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv("dgaa_mosfet_threshold_voltage_dataset.csv")

# Prepare data
X = df.drop("Threshold_Voltage_V", axis=1)
y = df["Threshold_Voltage_V"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Initialize Random Forest model
model = RandomForestRegressor(n_estimators=2000, random_state=42, 
                            max_leaf_nodes=1000, max_depth=100)

# Function to calculate bias and variance
def calculate_bias_variance(model, X_train, y_train, X_test, y_test, num_iterations=10):
    predictions = []
    
    for _ in range(num_iterations):
        # Fit the model
        model.fit(X_train, y_train)
        
        # Make predictions on test set
        y_pred = model.predict(X_test)
        predictions.append(y_pred)
    
    # Convert to numpy array
    predictions = np.array(predictions)
    
    # Calculate mean predictions across iterations
    mean_predictions = np.mean(predictions, axis=0)
    
    # Calculate bias (squared difference between mean prediction and true values)
    bias_squared = np.mean((mean_predictions - y_test)**2)
    
    # Calculate variance (variance of predictions around the mean prediction)
    variance = np.mean(np.var(predictions, axis=0))
    
    return bias_squared, variance

# Calculate learning curves to see how error changes with training size
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)

# Calculate MSE and variance for different training set sizes
train_mse = -train_scores.mean(axis=1)
test_mse = -test_scores.mean(axis=1)
test_var = test_scores.var(axis=1)

# Plot MSE vs Training Set Size
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_sizes, train_mse, 'o-', color="r", label="Training MSE")
plt.plot(train_sizes, test_mse, 'o-', color="g", label="Validation MSE")
plt.xlabel("Training Set Size")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curve - MSE vs Training Size")
plt.legend(loc="best")
plt.grid(True)

# Plot Variance vs Training Set Size
plt.subplot(1, 2, 2)
plt.plot(train_sizes, test_var, 'o-', color="b", label="Variance")
plt.xlabel("Training Set Size")
plt.ylabel("Variance of Error")
plt.title("Learning Curve - Variance vs Training Size")
plt.legend(loc="best")
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate bias and variance for the full model
bias_squared, variance = calculate_bias_variance(model, X_train, y_train, X_test, y_test)

print(f"\nBias²: {bias_squared:.6f}")
print(f"Variance: {variance:.6f}")
print(f"Total Error (Bias² + Variance): {bias_squared + variance:.6f}")

# Plot error distribution
y_pred = model.predict(X_test)
errors = y_test - y_pred

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(errors, bins=30, edgecolor='black')
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors")

plt.subplot(1, 2, 2)
plt.scatter(y_pred, errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()