"""
Cryptocurrency Liquidity Prediction - Main Pipeline
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and combine cryptocurrency data"""
    print("Loading data...")
    df1 = pd.read_csv('coin_gecko_2022-03-16.csv')
    df2 = pd.read_csv('coin_gecko_2022-03-17.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"Loaded {len(df)} records")
    return df

def preprocess_data(df):
    """Clean and preprocess the data"""
    print("Preprocessing data...")
    
    # Handle missing values
    df = df.dropna()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create additional features
    df['price_volatility_24h'] = abs(df['24h'])
    df['volume_to_mcap_ratio'] = df['24h_volume'] / df['mkt_cap']
    df['price_change_7d_abs'] = abs(df['7d'])
    df['liquidity_score'] = (df['24h_volume'] * df['volume_to_mcap_ratio']) / (1 + df['price_volatility_24h'])
    
    # Normalize numerical features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    numerical_cols = ['price', '24h_volume', 'mkt_cap', 'price_volatility_24h', 
                     'volume_to_mcap_ratio', 'price_change_7d_abs']
    
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, scaler

def train_models(df):
    """Train multiple ML models"""
    print("Training models...")
    
    # Features for prediction
    feature_cols = ['price', '24h_volume', 'mkt_cap', 'price_volatility_24h', 
                   'volume_to_mcap_ratio', 'price_change_7d_abs', '1h', '24h', '7d']
    
    X = df[feature_cols]
    y = df['liquidity_score']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        print(f"{name} - RMSE: {results[name]['rmse']:.4f}, MAE: {results[name]['mae']:.4f}, R²: {results[name]['r2']:.4f}")
    
    return results, X_test, y_test, feature_cols

def save_models(results, scaler):
    """Save trained models and scaler"""
    print("Saving models...")
    joblib.dump(results['Random Forest']['model'], 'liquidity_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Models saved successfully!")

if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    df_processed, scaler = preprocess_data(df)
    
    # Train models
    results, X_test, y_test, feature_cols = train_models(df_processed)
    
    # Save best model
    save_models(results, scaler)
    
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    for name, result in results.items():
        print(f"{name}: RMSE={result['rmse']:.4f}, MAE={result['mae']:.4f}, R²={result['r2']:.4f}")

