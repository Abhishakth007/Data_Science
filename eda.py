"""
Exploratory Data Analysis for Cryptocurrency Liquidity Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_and_prepare_data():
    """Load and prepare data for EDA"""
    df1 = pd.read_csv('coin_gecko_2022-03-16.csv')
    df2 = pd.read_csv('coin_gecko_2022-03-17.csv')
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Basic preprocessing
    df['date'] = pd.to_datetime(df['date'])
    df['price_volatility_24h'] = abs(df['24h'])
    df['volume_to_mcap_ratio'] = df['24h_volume'] / df['mkt_cap']
    df['liquidity_score'] = (df['24h_volume'] * df['volume_to_mcap_ratio']) / (1 + df['price_volatility_24h'])
    
    return df

def generate_eda_report(df):
    """Generate comprehensive EDA report"""
    print("=== CRYPTOCURRENCY LIQUIDITY EDA REPORT ===\n")
    
    # Dataset Overview
    print("1. DATASET OVERVIEW")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique cryptocurrencies: {df['coin'].nunique()}")
    print(f"Features: {list(df.columns)}")
    print()
    
    # Basic Statistics
    print("2. BASIC STATISTICS")
    print(df.describe())
    print()
    
    # Top cryptocurrencies by market cap
    print("3. TOP 10 CRYPTOCURRENCIES BY MARKET CAP")
    top_coins = df.groupby('coin')['mkt_cap'].mean().sort_values(ascending=False).head(10)
    print(top_coins)
    print()
    
    # Volume analysis
    print("4. VOLUME ANALYSIS")
    print(f"Average 24h volume: ${df['24h_volume'].mean():,.0f}")
    print(f"Median 24h volume: ${df['24h_volume'].median():,.0f}")
    print(f"Volume range: ${df['24h_volume'].min():,.0f} - ${df['24h_volume'].max():,.0f}")
    print()
    
    # Price volatility analysis
    print("5. PRICE VOLATILITY ANALYSIS")
    print(f"Average 24h price change: {df['24h'].mean():.4f}")
    print(f"Average 7d price change: {df['7d'].mean():.4f}")
    print(f"Most volatile coin (24h): {df.loc[df['24h'].idxmax(), 'coin']} ({df['24h'].max():.4f})")
    print(f"Least volatile coin (24h): {df.loc[df['24h'].idxmin(), 'coin']} ({df['24h'].min():.4f})")
    print()
    
    # Liquidity score analysis
    print("6. LIQUIDITY SCORE ANALYSIS")
    print(f"Average liquidity score: {df['liquidity_score'].mean():.4f}")
    print(f"Liquidity score range: {df['liquidity_score'].min():.4f} - {df['liquidity_score'].max():.4f}")
    print(f"Top 5 most liquid coins:")
    top_liquid = df.nlargest(5, 'liquidity_score')[['coin', 'liquidity_score']]
    print(top_liquid)
    print()

def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    # Set style
    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Price distribution
    plt.subplot(3, 3, 1)
    plt.hist(df['price'], bins=50, alpha=0.7, color='skyblue')
    plt.title('Price Distribution')
    plt.xlabel('Price (USD)')
    plt.ylabel('Frequency')
    
    # 2. Volume distribution
    plt.subplot(3, 3, 2)
    volume_log = np.log10(df['24h_volume'].replace(0, 1))  # Replace 0 with 1 to avoid log(0)
    plt.hist(volume_log, bins=50, alpha=0.7, color='lightgreen')
    plt.title('24h Volume Distribution (Log Scale)')
    plt.xlabel('Log10(Volume)')
    plt.ylabel('Frequency')
    
    # 3. Market cap vs Volume
    plt.subplot(3, 3, 3)
    # Filter out zero values for log scale
    valid_data = df[(df['mkt_cap'] > 0) & (df['24h_volume'] > 0)]
    plt.scatter(valid_data['mkt_cap'], valid_data['24h_volume'], alpha=0.6, color='orange')
    plt.title('Market Cap vs 24h Volume')
    plt.xlabel('Market Cap')
    plt.ylabel('24h Volume')
    plt.xscale('log')
    plt.yscale('log')
    
    # 4. Price changes
    plt.subplot(3, 3, 4)
    plt.hist(df['24h'], bins=30, alpha=0.7, color='red', label='24h')
    plt.hist(df['7d'], bins=30, alpha=0.7, color='blue', label='7d')
    plt.title('Price Changes Distribution')
    plt.xlabel('Price Change')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 5. Top coins by market cap
    plt.subplot(3, 3, 5)
    top_10 = df.groupby('coin')['mkt_cap'].mean().sort_values(ascending=False).head(10)
    top_10.plot(kind='bar', color='purple')
    plt.title('Top 10 Coins by Market Cap')
    plt.xlabel('Cryptocurrency')
    plt.ylabel('Market Cap')
    plt.xticks(rotation=45)
    
    # 6. Liquidity score distribution
    plt.subplot(3, 3, 6)
    plt.hist(df['liquidity_score'], bins=50, alpha=0.7, color='gold')
    plt.title('Liquidity Score Distribution')
    plt.xlabel('Liquidity Score')
    plt.ylabel('Frequency')
    
    # 7. Volume to Market Cap Ratio
    plt.subplot(3, 3, 7)
    plt.hist(df['volume_to_mcap_ratio'], bins=50, alpha=0.7, color='pink')
    plt.title('Volume to Market Cap Ratio')
    plt.xlabel('Ratio')
    plt.ylabel('Frequency')
    
    # 8. Price volatility
    plt.subplot(3, 3, 8)
    plt.hist(df['price_volatility_24h'], bins=50, alpha=0.7, color='cyan')
    plt.title('24h Price Volatility')
    plt.xlabel('Volatility')
    plt.ylabel('Frequency')
    
    # 9. Correlation heatmap
    plt.subplot(3, 3, 9)
    corr_cols = ['price', '24h_volume', 'mkt_cap', '24h', '7d', 'liquidity_score']
    corr_matrix = df[corr_cols].corr()
    plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Feature Correlation Matrix')
    plt.xticks(range(len(corr_cols)), corr_cols, rotation=45)
    plt.yticks(range(len(corr_cols)), corr_cols)
    
    plt.tight_layout()
    plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualizations saved as 'eda_visualizations.png'")

def create_interactive_plots(df):
    """Create interactive Plotly visualizations"""
    print("Creating interactive plots...")
    
    # Interactive scatter plot: Market Cap vs Volume
    fig1 = px.scatter(df, x='mkt_cap', y='24h_volume', 
                     color='liquidity_score', size='price',
                     hover_data=['coin', '24h', '7d'],
                     title='Market Cap vs Volume (Interactive)',
                     labels={'mkt_cap': 'Market Cap', '24h_volume': '24h Volume'})
    fig1.update_xaxes(type="log")
    fig1.update_yaxes(type="log")
    fig1.write_html('interactive_market_analysis.html')
    
    # Interactive time series
    fig2 = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Price Trends', 'Volume Trends'))
    
    for coin in df['coin'].unique()[:5]:  # Top 5 coins
        coin_data = df[df['coin'] == coin]
        fig2.add_trace(go.Scatter(x=coin_data['date'], y=coin_data['price'],
                                mode='lines', name=f'{coin} Price'), row=1, col=1)
        fig2.add_trace(go.Scatter(x=coin_data['date'], y=coin_data['24h_volume'],
                                mode='lines', name=f'{coin} Volume'), row=2, col=1)
    
    fig2.update_layout(title='Cryptocurrency Price and Volume Trends')
    fig2.write_html('interactive_trends.html')
    
    print("Interactive plots saved as HTML files")

if __name__ == "__main__":
    # Load data
    df = load_and_prepare_data()
    
    # Generate EDA report
    generate_eda_report(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Create interactive plots
    create_interactive_plots(df)
    
    print("EDA completed successfully!")
