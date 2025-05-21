import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor  # More accurate than XGBoost for this case
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import RFECV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import optuna  # For hyperparameter optimization
import joblib
import matplotlib.pyplot as plt

## Data Loading and Advanced Preparation
def load_and_preprocess():
    df = pd.read_csv(filename)
    
    # Advanced semiconductor physics features
    ε_ox = 3.9 * 8.854e-12  # SiO₂ permittivity
    df['Cox'] = ε_ox / (df['Oxide_Thickness_nm'] * 1e-9)
    df['Doping_log'] = np.log10(df['Channel_Doping_cm3'])
    df['Vds_norm'] = df['Drain_Voltage_V'] / df['Channel_Length_nm']
    df['Quantum_Effect'] = 1/(df['Channel_Thickness_nm']**2)
    df['Geometry_Factor'] = df['Channel_Length_nm'] * df['Channel_Thickness_nm'] / df['Inner_Gate_Radius_nm']
    
    # Remove potential outliers (top/bottom 1%)
    q_low = df['Threshold_Voltage_V'].quantile(0.01)
    q_hi = df['Threshold_Voltage_V'].quantile(0.99)
    df = df[(df['Threshold_Voltage_V'] > q_low) & (df['Threshold_Voltage_V'] < q_hi)]
    
    return df

## Feature Engineering
def create_features(df):
    X = df[[
        'Channel_Length_nm', 
        'Channel_Thickness_nm',
        'Oxide_Thickness_nm',
        'Inner_Gate_Radius_nm',
        'Drain_Voltage_V',
        'Cox',
        'Doping_log',
        'Vds_norm',
        'Quantum_Effect',
        'Geometry_Factor'
    ]]
    y = df['Threshold_Voltage_V']
    return X, y

## Hyperparameter Optimization
def optimize_hyperparameters(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1),
            'subsample': trial.suggest_float('subsample', 0.6, 1)
        }
        
        model = LGBMRegressor(**params, random_state=42, n_jobs=-1)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
        return np.mean(scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, timeout=3600)
    return study.best_params

## Main Pipeline
def build_model(X, y):
    # Feature selection
    selector = RFECV(
        estimator=LGBMRegressor(n_estimators=100),
        step=1,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        min_features_to_select=5
    )
    selector.fit(X, y)
    selected_features = X.columns[selector.support_]
    print(f"Selected features: {list(selected_features)}")
    
    # Hyperparameter tuning
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(X[selected_features], y)
    
    # Final model pipeline
    model = Pipeline([
        ('preprocessor', PowerTransformer()),
        ('regressor', LGBMRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    return model, selected_features

## Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Vth')
    plt.ylabel('Predicted Vth')
    plt.title(f'Prediction Performance (R² = {r2:.4f})')
    plt.grid()
    plt.savefig('prediction_plot.png', dpi=300)
    plt.close()
    
    return rmse, r2

## Main Execution
if __name__ == "__main__":
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_and_preprocess('DGAA_MOSFET_Vth_Dataset.csv')
    X, y = create_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    # Build model
    model, selected_features = build_model(X_train, y_train)
    
    # Train final model
    print("Training final model...")
    model.fit(X_train[selected_features], y_train)
    
    # Evaluate
    rmse, r2 = evaluate_model(model, X_test[selected_features], y_test)
    print(f"\nFinal Model Performance:")
    print(f"RMSE: {rmse:.6f} V")
    print(f"R² Score: {r2:.6f}")
    
    # Save model
    joblib.dump({
        'model': model,
        'features': list(selected_features),
        'performance': {'rmse': rmse, 'r2': r2}
    }, 'ultra_optimized_vth_predictor.joblib')
    
    # Example prediction
    example = pd.DataFrame([{
        'Channel_Length_nm': 32.5,
        'Channel_Thickness_nm': 5.2,
        'Oxide_Thickness_nm': 1.8,
        'Inner_Gate_Radius_nm': 3.1,
        'Drain_Voltage_V': 0.5,
        'Cox': 3.9*8.854e-12/(1.8e-9),
        'Doping_log': 17.2,
        'Vds_norm': 0.5/32.5,
        'Quantum_Effect': 1/(5.2**2),
        'Geometry_Factor': (32.5*5.2)/3.1
    }])
    
    pred = model.predict(example[selected_features])
    print(f"\nExample Prediction: {pred[0]:.4f} V")