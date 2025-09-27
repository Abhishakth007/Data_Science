# Cryptocurrency Liquidity Prediction System

A machine learning system to predict cryptocurrency liquidity levels for market stability assessment.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train Models**
   ```bash
   python main.py
   ```

3. **Run EDA Analysis**
   ```bash
   python eda.py
   ```

4. **Launch Web App**
   ```bash
   streamlit run app.py
   ```

5. **Open Browser**
   Navigate to `http://localhost:8501`

## 📊 Project Overview

This project predicts cryptocurrency liquidity levels using machine learning to help detect market instability risks early.

### Key Features
- **Liquidity Prediction**: Forecasts market liquidity using ML models
- **Risk Assessment**: Identifies potential market instability
- **Interactive Dashboard**: Web interface for real-time predictions
- **Data Visualization**: Comprehensive EDA and trend analysis

### Model Performance
- **Accuracy**: 72% (R² = 0.72)
- **RMSE**: 908,878,241
- **MAE**: 161,030,677

## 📁 Project Structure

```
├── main.py                    # Main ML pipeline
├── eda.py                     # Exploratory data analysis
├── app.py                     # Streamlit web app
├── requirements.txt           # Dependencies
├── HLD.md                     # High-level design
├── LLD.md                     # Low-level design
├── pipeline_architecture.md   # Pipeline documentation
├── eda_report.md              # EDA report
├── final_report.md            # Final project report
├── eda_visualizations.png     # EDA plots
├── interactive_market_analysis.html  # Interactive plots
├── interactive_trends.html    # Interactive trends
├── liquidity_model.pkl        # Trained model
├── scaler.pkl                 # Data scaler
└── *.csv                      # Data files
```

## 🔧 Technical Details

### Data Sources
- Historical cryptocurrency data (March 2022)
- 1,000 records across 506 cryptocurrencies
- Features: price, volume, market cap, price changes

### Features Engineered
- **Liquidity Score**: Primary target variable
- **Price Volatility**: 24h price change magnitude
- **Volume-to-Market-Cap Ratio**: Trading activity indicator
- **Price Change Patterns**: Short and long-term trends

### Models Used
- **Linear Regression**: Primary model (R² = 0.72)
- **Random Forest**: Alternative model (R² = 0.67)

## 📈 Usage

### Web Interface
1. Launch the Streamlit app
2. Input cryptocurrency parameters
3. Get liquidity predictions
4. View risk assessment
5. Explore data visualizations

### API Usage
```python
import joblib
import numpy as np

# Load model
model = joblib.load('liquidity_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare features
features = np.array([price, volume, mkt_cap, ...]).reshape(1, -1)
features_scaled = scaler.transform(features)

# Predict
liquidity_score = model.predict(features_scaled)[0]
```

## 📋 Requirements

- Python 3.13+
- 8GB RAM minimum
- 1GB storage
- Web browser

## 🎯 Business Impact

- **Risk Management**: Early detection of liquidity crises
- **Market Stability**: Maintain stable trading conditions
- **Decision Support**: Data-driven trading insights
- **Regulatory Compliance**: Market surveillance capabilities

## 🔮 Future Enhancements

- Real-time data integration
- Advanced deep learning models
- Social media sentiment analysis
- Cloud deployment
- API development

## 📞 Support

For questions or issues, refer to the documentation files:
- `HLD.md` - System architecture
- `LLD.md` - Technical implementation
- `pipeline_architecture.md` - Data flow
- `eda_report.md` - Data analysis
- `final_report.md` - Complete findings

---

**Status**: ✅ Production Ready
**Version**: 1.0.0
**Last Updated**: March 2024

