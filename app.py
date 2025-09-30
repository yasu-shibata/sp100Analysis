"""
Streamlit S&P100 Growth Analysis App
Analyzes S&P100 stocks by analyst recommendations with technical indicators
"""

import streamlit as st
import pandas as pd
import datetime as dt
import numpy as np
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

def calculate_macd_crossover_days(macd_series, signal_series):
    """Calculate days since MACD crossover"""
    days_since_crossover = []
    last_crossover_idx = None
    crossover_type = None
    
    for i in range(len(macd_series)):
        if i == 0:
            days_since_crossover.append(np.nan)
            continue
        
        prev_diff = macd_series.iloc[i-1] - signal_series.iloc[i-1]
        curr_diff = macd_series.iloc[i] - signal_series.iloc[i]
        
        if (prev_diff <= 0 and curr_diff > 0) or (prev_diff >= 0 and curr_diff < 0):
            last_crossover_idx = i
            days_since_crossover.append(0)
            crossover_type = "Bullish" if curr_diff > 0 else "Bearish"
        elif last_crossover_idx is not None:
            days_since_crossover.append(i - last_crossover_idx)
        else:
            days_since_crossover.append(np.nan)
    
    return pd.Series(days_since_crossover, index=macd_series.index), crossover_type

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
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    return df

def calculate_buy_signal_score(latest_data, close_price, sma_20, sma_50):
    """
    Calculate buy signal score based on multiple technical indicators
    Returns: score (0-10), signal strength, and reasons
    """
    buy_score = 0
    reasons = []
    
    rsi = latest_data.get('RSI', np.nan)
    macd = latest_data.get('MACD', np.nan)
    macd_signal = latest_data.get('MACD_Signal', np.nan)
    macd_hist = latest_data.get('MACD_Histogram', np.nan)
    bb_position = latest_data.get('BB_Position', np.nan)
    
    # RSI signals (0-3 points)
    if not np.isnan(rsi):
        if rsi < 30:
            buy_score += 3
            reasons.append("RSI oversold (<30)")
        elif rsi < 40:
            buy_score += 2
            reasons.append("RSI low (<40)")
        elif rsi > 70:
            buy_score -= 2
            reasons.append("RSI overbought (>70)")
    
    # MACD signals (0-3 points)
    if not np.isnan(macd) and not np.isnan(macd_signal):
        if macd > macd_signal and macd_hist > 0:
            buy_score += 2
            reasons.append("MACD bullish crossover")
        elif macd < macd_signal and macd_hist < 0:
            buy_score -= 1
            reasons.append("MACD bearish")
        
        # Recent crossover bonus
        crossover_days = latest_data.get('MACD_Crossover_Days', np.nan)
        if not np.isnan(crossover_days) and crossover_days <= 5 and macd > macd_signal:
            buy_score += 1
            reasons.append(f"Recent bullish cross ({int(crossover_days)}d)")
    
    # Trend signals (0-2 points)
    if not np.isnan(close_price) and not np.isnan(sma_20) and not np.isnan(sma_50):
        if close_price > sma_20 > sma_50:
            buy_score += 2
            reasons.append("Strong uptrend (SMA)")
        elif close_price < sma_20 < sma_50:
            buy_score -= 1
            reasons.append("Downtrend (SMA)")
    
    # Bollinger Band signals (0-2 points)
    if not np.isnan(bb_position):
        if bb_position < 0.2:
            buy_score += 2
            reasons.append("Near lower BB (oversold)")
        elif bb_position > 0.8:
            buy_score -= 1
            reasons.append("Near upper BB (overbought)")
    
    # Normalize score to 0-10
    buy_score = max(0, min(10, buy_score))
    
    # Determine signal strength
    if buy_score >= 7:
        signal = "Strong Buy"
        color = "green"
    elif buy_score >= 5:
        signal = "Buy"
        color = "lightgreen"
    elif buy_score >= 3:
        signal = "Neutral"
        color = "gray"
    elif buy_score >= 1:
        signal = "Sell"
        color = "orange"
    else:
        signal = "Strong Sell"
        color = "red"
    
    return buy_score, signal, color, reasons

@st.cache_data(ttl=3600)
def get_analyst_and_technical_data(ticker):
    """Get analyst estimates and technical indicators"""
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
        
        # Fetch historical data for technical analysis
        hist_data, error = fetch_stock_data(ticker, period="6mo")
        
        rsi = np.nan
        macd = np.nan
        macd_signal = np.nan
        macd_hist = np.nan
        macd_crossover_days = np.nan
        macd_crossover_type = "N/A"
        bb_position = np.nan
        sma_20 = np.nan
        sma_50 = np.nan
        buy_score = 0
        buy_signal = "N/A"
        buy_signal_color = "gray"
        buy_reasons = []
        
        if hist_data is not None:
            tech_data = calculate_technical_indicators(hist_data)
            latest = tech_data.iloc[-1]
            
            rsi = latest['RSI']
            macd = latest['MACD']
            macd_signal = latest['MACD_Signal']
            macd_hist = latest['MACD_Histogram']
            bb_position = latest['BB_Position']
            sma_20 = latest['SMA_20']
            sma_50 = latest['SMA_50']
            
            # Calculate crossover info
            crossover_series, cross_type = calculate_macd_crossover_days(
                tech_data['MACD'], 
                tech_data['MACD_Signal']
            )
            
            if not np.isnan(crossover_series.iloc[-1]):
                macd_crossover_days = int(crossover_series.iloc[-1])
                # Determine current crossover type
                if macd > macd_signal:
                    macd_crossover_type = "Bullish"
                else:
                    macd_crossover_type = "Bearish"
            
            # Calculate buy signal
            latest_dict = {
                'RSI': rsi,
                'MACD': macd,
                'MACD_Signal': macd_signal,
                'MACD_Histogram': macd_hist,
                'BB_Position': bb_position,
                'MACD_Crossover_Days': macd_crossover_days
            }
            buy_score, buy_signal, buy_signal_color, buy_reasons = calculate_buy_signal_score(
                latest_dict, current_price, sma_20, sma_50
            )
        
        return {
            'ticker': ticker,
            'company_name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'current_price': current_price,
            'target_price': target_price,
            'price_upside_pct': ((target_price - current_price) / current_price) * 100,
            'analyst_rec_score': analyst_rec_score,
            'next_earnings_date': next_earnings_date,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_hist,
            'macd_crossover_days': macd_crossover_days,
            'macd_crossover_type': macd_crossover_type,
            'buy_score': buy_score,
            'buy_signal': buy_signal,
            'buy_signal_color': buy_signal_color,
            'buy_reasons': ', '.join(buy_reasons) if buy_reasons else 'N/A'
        }
    except Exception:
        return None

def main():
    st.title("S&P100 Growth Analysis Dashboard")
    st.markdown("Analyze S&P100 stocks by analyst recommendations with technical indicators")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        top_n = st.slider("Number of Top Stocks", min_value=10, max_value=100, value=20, step=5)
        
        sort_by = st.selectbox(
            "Sort By",
            ["Buy Score", "Analyst Rec Score", "Price Upside %", "RSI", "MACD Histogram"]
        )
        
        filter_rsi = st.checkbox("Filter by RSI", value=False)
        if filter_rsi:
            rsi_min, rsi_max = st.slider("RSI Range", 0, 100, (30, 70))
        
        filter_buy_signal = st.checkbox("Filter by Buy Signal", value=False)
        if filter_buy_signal:
            min_buy_score = st.slider("Minimum Buy Score", 0, 10, 5)
        
        analyze_button = st.button("Start Analysis", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("""
        **Analysis Method:**
        - Fetches S&P100 constituents
        - Retrieves analyst recommendations
        - Calculates technical indicators (RSI, MACD)
        - Tracks MACD crossover information
        
        **Recommendation Score:**
        - 1.0 = Strong Buy
        - 2.0 = Buy
        - 3.0 = Hold
        - 4.0 = Sell
        - 5.0 = Strong Sell
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
        st.header("Comprehensive Stock Analysis")
        
        growth_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"Analyzing {ticker}... ({i+1}/{len(tickers)})")
            data = get_analyst_and_technical_data(ticker)
            if data:
                growth_data.append(data)
            progress_bar.progress((i + 1) / len(tickers))
            time.sleep(0.05)  # Rate limiting
        
        progress_bar.empty()
        status_text.empty()
        
        if not growth_data:
            st.error("No data could be retrieved")
            return
        
        # Create DataFrame
        df = pd.DataFrame(growth_data)
        
        # Apply filters
        if filter_rsi:
            df = df[(df['rsi'] >= rsi_min) & (df['rsi'] <= rsi_max)]
        
        if filter_buy_signal:
            df = df[df['buy_score'] >= min_buy_score]
        
        # Sort based on selection
        if sort_by == "Buy Score":
            df_sorted = df.sort_values('buy_score', ascending=False)
        elif sort_by == "Analyst Rec Score":
            df_sorted = df.sort_values('analyst_rec_score', ascending=True)
        elif sort_by == "Price Upside %":
            df_sorted = df.sort_values('price_upside_pct', ascending=False)
        elif sort_by == "RSI":
            df_sorted = df.sort_values('rsi', ascending=True)
        else:  # MACD Histogram
            df_sorted = df.sort_values('macd_histogram', ascending=False)
        
        df_sorted = df_sorted.head(top_n)
        
        # Display results
        st.subheader(f"Top {len(df_sorted)} Stocks")
        st.info(f"Sorted by: {sort_by} | Total stocks analyzed: {len(df)}")
        
        # Create display dataframe
        display_df = df_sorted[[
            'ticker', 'company_name', 'sector', 'current_price', 'target_price', 
            'price_upside_pct', 'analyst_rec_score', 'buy_score', 'buy_signal',
            'rsi', 'macd', 'macd_signal', 'macd_histogram', 'macd_crossover_days', 
            'macd_crossover_type', 'buy_reasons', 'next_earnings_date'
        ]].copy()
        
        display_df.columns = [
            'Ticker', 'Company', 'Sector', 'Price', 'Target', 'Upside %', 
            'Rec Score', 'Buy Score', 'Buy Signal', 'RSI', 'MACD', 'Signal', 
            'MACD Hist', 'Cross Days', 'Cross Type', 'Buy Reasons', 'Next Earnings'
        ]
        
        # Format numeric columns
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
        display_df['Target'] = display_df['Target'].apply(lambda x: f"${x:.2f}" if not np.isnan(x) else "N/A")
        display_df['Upside %'] = display_df['Upside %'].apply(lambda x: f"{x:.1f}%" if not np.isnan(x) else "N/A")
        display_df['Rec Score'] = display_df['Rec Score'].apply(lambda x: f"{x:.2f}" if not np.isnan(x) else "N/A")
        display_df['Buy Score'] = display_df['Buy Score'].apply(lambda x: f"{x:.1f}" if not np.isnan(x) else "N/A")
        display_df['RSI'] = display_df['RSI'].apply(lambda x: f"{x:.1f}" if not np.isnan(x) else "N/A")
        display_df['MACD'] = display_df['MACD'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        display_df['Signal'] = display_df['Signal'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        display_df['MACD Hist'] = display_df['MACD Hist'].apply(lambda x: f"{x:.4f}" if not np.isnan(x) else "N/A")
        display_df['Cross Days'] = display_df['Cross Days'].apply(lambda x: f"{int(x)}" if not np.isnan(x) else "N/A")
        
        # Apply color coding
        def color_rec_score(val):
            try:
                score = float(val)
                if score <= 2.0:
                    return 'background-color: #90EE90'
                elif score <= 3.0:
                    return 'background-color: #FFFFE0'
                else:
                    return 'background-color: #FFB6C6'
            except:
                return ''
        
        def color_rsi(val):
            try:
                rsi = float(val)
                if rsi < 30:
                    return 'background-color: #90EE90'
                elif rsi > 70:
                    return 'background-color: #FFB6C6'
                else:
                    return ''
            except:
                return ''
        
        def color_crossover_type(val):
            if val == 'Bullish':
                return 'background-color: #90EE90'
            elif val == 'Bearish':
                return 'background-color: #FFB6C6'
            else:
                return ''
        
        def color_buy_signal(val):
            if val == 'Strong Buy':
                return 'background-color: #006400; color: white'
            elif val == 'Buy':
                return 'background-color: #90EE90'
            elif val == 'Neutral':
                return 'background-color: #D3D3D3'
            elif val == 'Sell':
                return 'background-color: #FFA500'
            elif val == 'Strong Sell':
                return 'background-color: #FF0000; color: white'
            else:
                return ''
        
        styled_df = display_df.style\
            .applymap(color_rec_score, subset=['Rec Score'])\
            .applymap(color_rsi, subset=['RSI'])\
            .applymap(color_crossover_type, subset=['Cross Type'])\
            .applymap(color_buy_signal, subset=['Buy Signal'])
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = df_sorted.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name=f"sp100_analysis_{dt.datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.markdown("---")
        st.subheader("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_rec_score = df_sorted['analyst_rec_score'].mean()
            st.metric("Avg Rec Score", f"{avg_rec_score:.2f}")
        
        with col2:
            avg_upside = df_sorted['price_upside_pct'].mean()
            st.metric("Avg Upside", f"{avg_upside:.1f}%")
        
        with col3:
            avg_rsi = df_sorted['rsi'].mean()
            st.metric("Avg RSI", f"{avg_rsi:.1f}")
        
        with col4:
            bullish_count = len(df_sorted[df_sorted['macd_crossover_type'] == 'Bullish'])
            st.metric("Bullish MACD", f"{bullish_count}/{len(df_sorted)}")
        
        # Buy signal breakdown
        st.markdown("---")
        st.subheader("Buy Signal Distribution")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        signal_counts = df_sorted['buy_signal'].value_counts()
        
        with col1:
            strong_buy = signal_counts.get('Strong Buy', 0)
            st.metric("Strong Buy", strong_buy, help="Buy Score >= 7")
        
        with col2:
            buy = signal_counts.get('Buy', 0)
            st.metric("Buy", buy, help="Buy Score 5-6")
        
        with col3:
            neutral = signal_counts.get('Neutral', 0)
            st.metric("Neutral", neutral, help="Buy Score 3-4")
        
        with col4:
            sell = signal_counts.get('Sell', 0)
            st.metric("Sell", sell, help="Buy Score 1-2")
        
        with col5:
            strong_sell = signal_counts.get('Strong Sell', 0)
            st.metric("Strong Sell", strong_sell, help="Buy Score 0")
        
        # Sector breakdown
        st.markdown("---")
        st.subheader("Sector Distribution")
        sector_counts = df_sorted['sector'].value_counts()
        
        sector_df = pd.DataFrame({
            'Sector': sector_counts.index,
            'Count': sector_counts.values
        })
        
        st.dataframe(sector_df, use_container_width=True, hide_index=True)
        
        st.success("Analysis complete!")
    
    else:
        # Welcome screen
        st.info("Configure your analysis in the sidebar and click 'Start Analysis'")
        
        st.markdown("""
        ### Features
        
        - Fetches real-time S&P100 constituent list
        - Analyst recommendation scores and target prices
        - Technical indicators: RSI, MACD, MACD Signal, MACD Histogram
        - MACD crossover tracking with days since last crossover
        - Bullish/Bearish crossover identification
        - Next earnings dates
        - Multiple sorting options
        - RSI filtering
        - CSV export functionality
        
        ### Understanding the Metrics
        
        **Analyst Recommendation Score**
        - Lower is better (1.0 = Strong Buy, 5.0 = Strong Sell)
        
        **RSI (Relative Strength Index)**
        - < 30: Oversold (potential buy)
        - > 70: Overbought (potential sell)
        - 30-70: Neutral
        
        **MACD (Moving Average Convergence Divergence)**
        - Bullish: MACD > Signal (upward momentum)
        - Bearish: MACD < Signal (downward momentum)
        - Histogram: Difference between MACD and Signal
        
        **MACD Crossover**
        - Days Since Crossover: Time elapsed since last crossover
        - Cross Type: Bullish (buy signal) or Bearish (sell signal)
        
        **Buy Score (0-10)**
        - Combines multiple technical indicators
        - **Strong Buy (7-10)**: Multiple bullish signals aligned
        - **Buy (5-6)**: Generally positive momentum
        - **Neutral (3-4)**: Mixed signals
        - **Sell (1-2)**: Bearish indicators present
        - **Strong Sell (0)**: Multiple bearish signals
        
        **Buy Score Calculation:**
        - RSI oversold (<30): +3 points
        - MACD bullish crossover: +2 points
        - Recent bullish cross (≤5 days): +1 point
        - Strong uptrend (SMA): +2 points
        - Near lower Bollinger Band: +2 points
        - (Negative points for bearish signals)
        """)

if __name__ == "__main__":
    main()