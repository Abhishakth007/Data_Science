# Low-Level Design (LLD) Document
## Cryptocurrency Liquidity Prediction System

### 1. Module Structure

```
crypto_liquidity_predictor/
├── main.py                 # Main training pipeline
├── eda.py                  # Exploratory data analysis
├── app.py                  # Streamlit web application
├── requirements.txt        # Dependencies
├── HLD.md                 # High-level design
├── LLD.md                 # Low-level design
├── pipeline_architecture.md # Pipeline documentation
├── eda_report.md          # EDA report
├── final_report.md        # Final project report
└── models/                # Saved models
    ├── liquidity_model.pkl
    └── scaler.pkl
```

### 2. Detailed Component Design

#### 2.1 Main Pipeline (`main.py`)

**Purpose**: Core ML pipeline for data processing, model training, and evaluation

**Key Functions**:
```python
def load_data():
    """Load and combine cryptocurrency data from CSV files"""
    - Input: CSV file paths
    - Output: Combined pandas DataFrame
    - Error handling: File not found, data format issues

def preprocess_data(df):
    """Clean and preprocess the data"""
    - Handle missing values
    - Create derived features (liquidity_score, volatility, ratios)
    - Normalize numerical features using StandardScaler
    - Return processed DataFrame and fitted scaler

def train_models(df):
    """Train multiple ML models"""
    - Features: ['price', '24h_volume', 'mkt_cap', 'price_volatility_24h', 
                 'volume_to_mcap_ratio', 'price_change_7d_abs', '1h', '24h', '7d']
    - Target: 'liquidity_score'
    - Models: LinearRegression, RandomForestRegressor
    - Evaluation: RMSE, MAE, R²
    - Return: Model results, test data, feature names

def save_models(results, scaler):
    """Save trained models and scaler"""
    - Save best model (Random Forest) as 'liquidity_model.pkl'
    - Save scaler as 'scaler.pkl'
    - Use joblib for efficient serialization
```

#### 2.2 EDA Module (`eda.py`)

**Purpose**: Comprehensive exploratory data analysis and visualization

**Key Functions**:
```python
def load_and_prepare_data():
    """Load data and create basic features for EDA"""
    - Load CSV files
    - Create liquidity_score and derived features
    - Return processed DataFrame

def generate_eda_report(df):
    """Generate comprehensive EDA report"""
    - Dataset overview and statistics
    - Top cryptocurrencies analysis
    - Volume and volatility analysis
    - Liquidity score distribution
    - Print formatted report to console

def create_visualizations(df):
    """Create comprehensive visualizations"""
    - 9 subplot visualization grid
    - Price, volume, market cap distributions
    - Correlation heatmap
    - Top coins analysis
    - Save as 'eda_visualizations.png'

def create_interactive_plots(df):
    """Create interactive Plotly visualizations"""
    - Interactive scatter plots
    - Time series analysis
    - Save as HTML files for web viewing
```

#### 2.3 Web Application (`app.py`)

**Purpose**: Streamlit-based web interface for predictions and data exploration

**Key Components**:
```python
@st.cache_data
def load_models():
    """Load trained models with caching"""
    - Load model and scaler from pickle files
    - Error handling for missing models
    - Return model and scaler objects

def predict_liquidity(coin_data, model, scaler):
    """Predict liquidity for given coin data"""
    - Prepare feature vector
    - Apply scaling transformation
    - Return liquidity prediction

def main():
    """Main Streamlit application"""
    - Page configuration and layout
    - Input form for prediction parameters
    - Results display with interpretation
    - Data exploration section
    - Interactive visualizations
```

### 3. Data Schema

#### 3.1 Input Data Schema
```python
{
    'coin': str,           # Cryptocurrency name
    'symbol': str,         # Trading symbol
    'price': float,        # Current price in USD
    '1h': float,          # 1-hour price change %
    '24h': float,         # 24-hour price change %
    '7d': float,          # 7-day price change %
    '24h_volume': float,  # 24-hour trading volume
    'mkt_cap': float,     # Market capitalization
    'date': str           # Date (YYYY-MM-DD)
}
```

#### 3.2 Derived Features Schema
```python
{
    'price_volatility_24h': float,      # abs(24h price change)
    'volume_to_mcap_ratio': float,      # 24h_volume / mkt_cap
    'price_change_7d_abs': float,       # abs(7d price change)
    'liquidity_score': float            # Primary target variable
}
```

### 4. Model Architecture

#### 4.1 Feature Engineering
```python
# Liquidity Score Formula
liquidity_score = (24h_volume * volume_to_mcap_ratio) / (1 + price_volatility_24h)

# Feature Selection
features = ['price', '24h_volume', 'mkt_cap', 'price_volatility_24h', 
           'volume_to_mcap_ratio', 'price_change_7d_abs', '1h', '24h', '7d']
```

#### 4.2 Model Configuration
```python
# Random Forest Regressor
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1
)

# Linear Regression
LinearRegression(
    fit_intercept=True,
    normalize=False
)
```

#### 4.3 Evaluation Metrics
```python
# Performance Metrics
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error  
- R²: Coefficient of Determination
```

### 5. Error Handling

#### 5.1 Data Loading Errors
```python
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File {file_path} not found")
except pd.errors.EmptyDataError:
    print(f"Error: File {file_path} is empty")
```

#### 5.2 Model Loading Errors
```python
try:
    model = joblib.load('liquidity_model.pkl')
except FileNotFoundError:
    st.error("Models not found! Please run main.py first")
    return None
```

#### 5.3 Prediction Errors
```python
try:
    prediction = model.predict(features_scaled)
except Exception as e:
    st.error(f"Prediction failed: {str(e)}")
    return None
```

### 6. Performance Optimization

#### 6.1 Caching Strategy
- Streamlit `@st.cache_data` for model loading
- Avoid redundant data processing
- Efficient memory usage

#### 6.2 Data Processing
- Vectorized operations using NumPy/Pandas
- Efficient feature engineering
- Minimal data copying

#### 6.3 Model Persistence
- Joblib for fast model serialization
- Compressed pickle files
- Quick model loading

### 7. Testing Strategy

#### 7.1 Unit Tests
- Individual function testing
- Data validation tests
- Model prediction tests

#### 7.2 Integration Tests
- End-to-end pipeline testing
- Web application testing
- Model performance validation

#### 7.3 Data Quality Tests
- Missing value detection
- Data type validation
- Range checking for numerical features

