"""
Streamlit S&P100 Growth Analysis App
Analyzes S&P100 stocks by analyst recommendations with technical analysis
"""

import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import requests
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="S&P100 Growth Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['font.family'] = 'sans-serif'

# Cache functions
@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sp100_tickers():
    """Fetch S&P100 tickers from Wikipedia"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    url = "https://en.wikipedia.org/wiki/S%26P_100"
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(response.text, attrs={'id': 'constituents'})
        
        if len(tables) == 0:
            return None, "No tables found on the page"
        
        df = tables[0]
        
        if 'Symbol' in df.columns:
            tickers = df['Symbol'].tolist()
            return tickers, None
        else:
            return None, "'Symbol' column not found"
    
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, period="6mo"):
    """Fetch stock data"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if not data.empty:
            return data.dropna(), None
        return None, "No data available"
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def get_analyst_estimates(ticker):
    """Get analyst estimates and recommendations"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or len(info) < 5:
            return None
        
        # Get next earnings date
        next_earnings_date = info.get('nextEpsReportDate')
        if not next_earnings_date:
            try:
                dates_df = stock.earnings_dates
                if not dates_df.empty:
                    today = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                    future_dates = dates_df[dates_df.index >= today]
                    if not future_dates.empty:
                        next_earnings_date = future_dates.index.min().tz_localize(None).strftime('%Y-%m-%d')
            except:
                next_earnings_date = "N/A"
        else:
            next_earnings_date = dt.datetime.fromtimestamp(next_earnings_date).strftime('%Y-%m-%d')
        
        analyst_rec_score = info.get('recommendationMean')
        target_price = info.get('targetMeanPrice')
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        
        if not all([target_price, current_price, current_price > 0, analyst_rec_score]):
            return None
        
        return {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'current_price': current_price,
            'target_price': target_price,
            'price_upside_pct': ((target_price - current_price) / current_price) * 100,
            'earnings_growth': info.get('earningsGrowth') or info.get('earningsQuarterlyGrowth'),
            'revenue_growth': info.get('revenueGrowth') or info.get('revenueQuarterlyGrowth'),
            'peg_ratio': info.get('pegRatio'),
            'analyst_rec_score': analyst_rec_score,
            'next_earnings_date': next_earnings_date,
        }
    except Exception:
        return None

def calculate_macd_crossover_days(macd_series, signal_series):
    """Calculate days since MACD crossover"""
    days_since_crossover = []
    last_crossover_idx = None
    
    for i in range(len(macd_series)):
        if i == 0:
            days_since_crossover.append(np.nan)
            continue
        
        prev_diff = macd_series.iloc[i-1] - signal_series.iloc[i-1]
        curr_diff = macd_series.iloc[i] - signal_series.iloc[i]
        
        if (prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0):
            last_crossover_idx = i
            days_since_crossover.append(0)
        elif last_crossover_idx is not None:
            days_since_crossover.append(i - last_crossover_idx)
        else:
            days_since_crossover.append(np.nan)
    
    return pd.Series(days_since_crossover, index=macd_series.index)

def calculate_technical_indicators(data):
    """Calculate technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # EMA & MACD
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    df['MACD_Crossover_Days'] = calculate_macd_crossover_days(df['MACD'], df['MACD_Signal'])
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    
    return df

def plot_technical_chart(data, symbol, company_name):
    """Create technical analysis chart"""
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10),
                                              gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Price chart
    ax1.plot(data.index, data['Close'], linewidth=2, label='Close', color='black')
    ax1.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7, color='blue')
    ax1.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7, color='red')
    ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1, color='gray')
    ax1.set_title(f"{symbol} - {company_name}", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Volume
    colors = ['green' if c >= o else 'red' for c, o in zip(data['Close'], data['Open'])]
    ax2.bar(data.index, data['Volume'], color=colors, alpha=0.7)
    ax2.plot(data.index, data['Volume_MA'], color='orange', linewidth=2, label='MA 20')
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # RSI
    ax3.plot(data.index, data['RSI'], color='purple', linewidth=2)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(y=30, color='blue', linestyle='--', alpha=0.7)
    ax3.fill_between(data.index, 30, 70, alpha=0.1, color='gray')
    ax3.set_ylabel('RSI')
    ax3.set_ylim(0, 100)
    ax3.grid(True, alpha=0.3)
    
    # MACD
    ax4.plot(data.index, data['MACD'], color='blue', linewidth=2, label='MACD')
    ax4.plot(data.index, data['MACD_Signal'], color='red', linewidth=2, label='Signal')
    colors_macd = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
    ax4.bar(data.index, data['MACD_Histogram'], alpha=0.3, color=colors_macd)
    ax4.axhline(y=0, color='black', alpha=0.5)
    ax4.set_ylabel('MACD')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.title("S&P100 Growth Analysis Dashboard")
    st.markdown("Analyze S&P100 stocks by analyst recommendations with technical analysis")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Analysis options
        top_n = st.slider("Number of Top Stocks", min_value=5, max_value=30, value=20, step=5)
        
        technical_top_n = st.slider("Technical Analysis (Top N)", min_value=3, max_value=15, value=10, step=1)
        
        include_spy = st.checkbox("Include SPY in Technical Analysis", value=True)
        
        analyze_button = st.button("Start Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        **Analysis Method:**
        - Fetches S&P100 constituents
        - Ranks by analyst recommendation score
        - Lower score = Stronger Buy recommendation
        - Performs technical analysis on top stocks
        """)
    
    # Main content
    if analyze_button:
        # Step 1: Get S&P100 tickers
        with st.spinner("Fetching S&P100 constituents..."):
            tickers, error = get_sp100_tickers()
            
            if error:
                st.error(f"Failed to fetch S&P100 tickers: {error}")
                return
            
            st.success(f"Fetched {len(tickers)} S&P100 stocks")
        
        # Step 2: Analyze stocks
        st.markdown("---")
        st.header("Analyst Recommendation Analysis")
        
        growth_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"Analyzing {ticker}... ({i+1}/{len(tickers)})")
            data = get_analyst_estimates(ticker)
            if data:
                growth_data.append(data)
            progress_bar.progress((i + 1) / len(tickers))
            time.sleep(0.05)  # Rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        if not growth_data:
            st.error("No analyst data could be retrieved")
            return
        
        # Create DataFrame and sort
        df = pd.DataFrame(growth_data)
        df_sorted = df.sort_values('analyst_rec_score', ascending=True).head(top_n)
        
        # Display results
        st.subheader(f"Top {top_n} Stocks by Analyst Recommendation")
        st.info("Lower recommendation score = Stronger Buy recommendation (1.0 = Strong Buy, 5.0 = Strong Sell)")
        
        # Create display dataframe
        display_df = df_sorted[['ticker', 'company_name', 'sector', 'current_price', 
                                'target_price', 'price_upside_pct', 'analyst_rec_score', 
                                'next_earnings_date']].copy()
        
        display_df.columns = ['Ticker', 'Company', 'Sector', 'Price ($)', 'Target ($)', 
                             'Upside (%)', 'Rec Score', 'Next Earnings']
        
        display_df['Price ($)'] = display_df['Price ($)'].apply(lambda x: f"${x:.2f}")
        display_df['Target ($)'] = display_df['Target ($)'].apply(lambda x: f"${x:.2f}")
        display_df['Upside (%)'] = display_df['Upside (%)'].apply(lambda x: f"{x:.1f}%")
        display_df['Rec Score'] = display_df['Rec Score'].apply(lambda x: f"{x:.2f}")
        
        # Color code recommendation score
        def color_rec_score(val):
            try:
                score = float(val)
                if score <= 2.0:
                    return 'background-color: #90EE90'  # Light green
                elif score <= 3.0:
                    return 'background-color: #FFFFE0'  # Light yellow
                else:
                    return 'background-color: #FFB6C6'  # Light red
            except:
                return ''
        
        st.dataframe(
            display_df.style.applymap(color_rec_score, subset=['Rec Score']),
            use_container_width=True,
            hide_index=True
        )
        
        # Sector distribution
        st.subheader("Sector Distribution")
        sector_counts = df_sorted['sector'].value_counts()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sector_counts.plot(kind='barh', ax=ax, color='steelblue')
            ax.set_xlabel('Number of Stocks')
            ax.set_title('Top Stocks by Sector')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.write("**Sector Breakdown:**")
            for sector, count in sector_counts.items():
                st.write(f"- {sector}: {count}")
        
        # Technical Analysis
        st.markdown("---")
        st.header(f"Technical Analysis - Top {technical_top_n} Stocks")
        
        top_stocks = df_sorted.head(technical_top_n)['ticker'].tolist()
        
        if include_spy:
            top_stocks.append('SPY')
        
        st.info(f"Analyzing: {', '.join(top_stocks)}")
        
        # Fetch and analyze technical data
        technical_data = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(top_stocks):
            status_text.text(f"Fetching technical data for {symbol}... ({i+1}/{len(top_stocks)})")
            data, error = fetch_stock_data(symbol, period="6mo")
            if data is not None:
                technical_data[symbol] = calculate_technical_indicators(data)
            progress_bar.progress((i + 1) / len(top_stocks))
        
        progress_bar.empty()
        status_text.empty()
        
        if not technical_data:
            st.error("Could not fetch technical data")
            return
        
        # Comparison Chart
        st.subheader("Price Performance Comparison")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Normalized price
        for symbol in technical_data:
            data = technical_data[symbol]
            normalized = data['Close'] / data['Close'].iloc[0]
            ax1.plot(data.index, normalized, label=symbol, linewidth=2)
        
        ax1.set_title('Normalized Price Comparison (Base = 1.0)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Normalized Price')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # RSI comparison
        for symbol in technical_data:
            data = technical_data[symbol]
            ax2.plot(data.index, data['RSI'], label=symbol, alpha=0.8)
        
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=30, color='blue', linestyle='--', alpha=0.7)
        ax2.set_title('RSI Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Individual stock charts
        st.subheader("Individual Stock Analysis")
        
        for symbol in top_stocks:
            if symbol not in technical_data:
                continue
            
            data = technical_data[symbol]
            latest = data.iloc[-1]
            
            # Get company info
            company_name = symbol
            for stock_data in growth_data:
                if stock_data['ticker'] == symbol:
                    company_name = stock_data['company_name']
                    break
            
            with st.expander(f"{symbol} - {company_name}"):
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Price", f"${latest['Close']:.2f}")
                
                with col2:
                    st.metric("RSI", f"{latest['RSI']:.1f}")
                
                with col3:
                    st.metric("MACD", f"{latest['MACD']:.4f}")
                
                with col4:
                    if not np.isnan(latest['MACD_Crossover_Days']):
                        cross_type = "Bull" if latest['MACD'] > latest['MACD_Signal'] else "Bear"
                        st.metric("MACD Cross", f"{int(latest['MACD_Crossover_Days'])}d", cross_type)
                    else:
                        st.metric("MACD Cross", "N/A")
                
                # Chart
                fig = plot_technical_chart(data, symbol, company_name)
                st.pyplot(fig)
                plt.close()
        
        st.success("Analysis complete!")
    
    else:
        # Welcome screen
        st.info("Configure your analysis in the sidebar and click 'Start Analysis'")
        
        st.markdown("""
        ### How It Works
        
        1. **Fetch S&P100 Stocks**: Automatically retrieves the current S&P100 constituents
        2. **Analyst Recommendations**: Ranks stocks by analyst recommendation scores
        3. **Technical Analysis**: Performs comprehensive technical analysis on top-ranked stocks
        4. **Visualization**: Creates comparison charts and individual stock analysis
        
        ### Key Metrics
        
        - **Recommendation Score**: 1.0 (Strong Buy) to 5.0 (Strong Sell)
        - **Price Upside**: Potential gain based on analyst target prices
        - **MACD Crossover**: Days since last bullish/bearish crossover
        - **RSI**: Relative Strength Index for momentum analysis
        
        ### Analysis Features
        
        - Sector distribution analysis
        - Multi-stock performance comparison
        - Individual technical charts with MACD, RSI, Bollinger Bands
        - Volume analysis
        - Optional SPY benchmark comparison
        """)

if __name__ == "__main__":
    main()