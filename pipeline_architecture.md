# Pipeline Architecture Document
## Cryptocurrency Liquidity Prediction System

### 1. Pipeline Overview

The cryptocurrency liquidity prediction pipeline is designed as a modular, end-to-end system that processes raw cryptocurrency data and produces liquidity predictions through a web interface.

### 2. Pipeline Flow Diagram

```
┌─────────────────┐
│   Raw Data      │
│   (CSV Files)   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Data Loading   │
│  & Validation   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Data Cleaning  │
│  & Preprocessing│
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Feature         │
│ Engineering     │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Data           │
│  Normalization  │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Train/Test     │
│  Split          │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Model Training │
│  & Evaluation   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Model          │
│  Persistence    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│  Web Interface  │
│  (Streamlit)    │
└─────────────────┘
```

### 3. Detailed Pipeline Components

#### 3.1 Data Ingestion Layer
**Input**: CSV files containing historical cryptocurrency data
**Process**:
- Load multiple CSV files
- Combine datasets
- Basic data validation
- Schema verification

**Code Location**: `main.py::load_data()`

#### 3.2 Data Preprocessing Layer
**Input**: Raw combined dataset
**Process**:
- Handle missing values
- Data type conversion
- Date parsing
- Outlier detection

**Code Location**: `main.py::preprocess_data()`

#### 3.3 Feature Engineering Layer
**Input**: Cleaned dataset
**Process**:
- Create liquidity score (target variable)
- Calculate price volatility indicators
- Compute volume-to-market-cap ratios
- Generate price change patterns

**Features Created**:
```python
# Target Variable
liquidity_score = (24h_volume * volume_to_mcap_ratio) / (1 + price_volatility_24h)

# Derived Features
price_volatility_24h = abs(24h_price_change)
volume_to_mcap_ratio = 24h_volume / market_cap
price_change_7d_abs = abs(7d_price_change)
```

#### 3.4 Data Normalization Layer
**Input**: Engineered features
**Process**:
- Apply StandardScaler to numerical features
- Preserve feature relationships
- Enable model convergence

**Code Location**: `main.py::preprocess_data()`

#### 3.5 Model Training Layer
**Input**: Normalized features and target
**Process**:
- 80/20 train-test split
- Train multiple models (Linear Regression, Random Forest)
- Cross-validation
- Hyperparameter tuning

**Models**:
- Linear Regression
- Random Forest Regressor (100 estimators)

#### 3.6 Model Evaluation Layer
**Input**: Trained models and test data
**Process**:
- Generate predictions
- Calculate performance metrics
- Compare model performance
- Select best model

**Metrics**:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (Coefficient of Determination)

#### 3.7 Model Persistence Layer
**Input**: Trained models and scaler
**Process**:
- Serialize best model using joblib
- Save scaler for preprocessing
- Version control for model updates

**Output Files**:
- `liquidity_model.pkl` (Random Forest model)
- `scaler.pkl` (StandardScaler)

#### 3.8 Web Interface Layer
**Input**: User parameters and saved models
**Process**:
- Load trained models
- Accept user input
- Preprocess input data
- Generate predictions
- Display results with interpretation

**Code Location**: `app.py`

### 4. Data Flow Specifications

#### 4.1 Data Schema Evolution

**Raw Data Schema**:
```python
{
    'coin': str,
    'symbol': str,
    'price': float,
    '1h': float,
    '24h': float,
    '7d': float,
    '24h_volume': float,
    'mkt_cap': float,
    'date': str
}
```

**Processed Data Schema**:
```python
{
    # Original features (normalized)
    'price': float,
    '24h_volume': float,
    'mkt_cap': float,
    '1h': float,
    '24h': float,
    '7d': float,
    
    # Engineered features
    'price_volatility_24h': float,
    'volume_to_mcap_ratio': float,
    'price_change_7d_abs': float,
    'liquidity_score': float
}
```

#### 4.2 Pipeline Configuration

**Data Split**:
- Training: 80%
- Testing: 20%
- Random State: 42 (reproducibility)

**Feature Scaling**:
- Method: StandardScaler
- Apply to: All numerical features
- Preserve: Feature distributions

**Model Selection**:
- Primary: Random Forest Regressor
- Fallback: Linear Regression
- Selection Criteria: R² score

### 5. Error Handling & Validation

#### 5.1 Data Validation
```python
# Missing value handling
df = df.dropna()

# Data type validation
df['date'] = pd.to_datetime(df['date'])

# Range validation
assert df['price'].min() > 0, "Price must be positive"
assert df['24h_volume'].min() >= 0, "Volume must be non-negative"
```

#### 5.2 Model Validation
```python
# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)

# Performance thresholds
assert r2_score > 0.5, "Model performance below threshold"
assert rmse < 1000000, "RMSE too high"
```

#### 5.3 Runtime Validation
```python
# Model loading validation
try:
    model = joblib.load('liquidity_model.pkl')
except FileNotFoundError:
    raise Exception("Model file not found")

# Prediction validation
assert prediction >= 0, "Liquidity score must be non-negative"
```

### 6. Performance Optimization

#### 6.1 Data Processing
- Vectorized operations using NumPy/Pandas
- Efficient memory usage
- Minimal data copying

#### 6.2 Model Training
- Parallel processing for Random Forest
- Efficient feature selection
- Optimized hyperparameters

#### 6.3 Web Interface
- Model caching with Streamlit
- Efficient data loading
- Responsive UI components

### 7. Monitoring & Logging

#### 7.1 Pipeline Monitoring
- Data quality metrics
- Model performance tracking
- Error rate monitoring

#### 7.2 Logging Strategy
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline steps
logger.info("Loading data...")
logger.info("Preprocessing data...")
logger.info("Training models...")
logger.info("Saving models...")
```

### 8. Deployment Architecture

#### 8.1 Local Deployment
- Single-machine setup
- Streamlit web server
- File-based model storage

#### 8.2 Scalability Considerations
- Containerization ready
- API endpoint development
- Database integration potential
- Cloud deployment support

### 9. Pipeline Maintenance

#### 9.1 Model Updates
- Retraining schedule
- Performance monitoring
- A/B testing framework

#### 9.2 Data Updates
- Automated data ingestion
- Schema evolution handling
- Data quality monitoring

#### 9.3 Code Maintenance
- Version control
- Testing framework
- Documentation updates

