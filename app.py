import streamlit as st
import pandas as pd
import os
import yfinance as yf
from openai import OpenAI
import requests
from datetime import datetime
import plotly.graph_objects as go

# --- 1. CONFIGURATION & SECRETS ---
# Accessing API keys from Streamlit Cloud Secrets
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

# Initializing OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Filenames for CSV storage
PORTFOLIO_FILE = "my_portfolio.csv"
MOCK_FILE = "mock_portfolio.csv"
LOG_FILE = "advisor_log.csv"

# --- 2. DATA STORAGE ENGINE ---
def load_data(file, columns):
    """Loads a CSV file into a DataFrame; handles missing files or errors."""
    if os.path.exists(file): 
        try:
            return pd.read_csv(file)
        except Exception:
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def save_data(df, file):
    """Saves the DataFrame back to a CSV file."""
    df.to_csv(file, index=False)

def log_recommendation(symbol, name, analysis, price):
    """Records the AI analysis into the historical log automatically."""
    log_df = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    
    new_entry = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Symbol": symbol,
        "Name": name,
        "Price_At_Rec": round(price, 2),
        "Analysis": analysis
    }])
    
    updated_log = pd.concat([log_df, new_entry], ignore_index=True)
    save_data(updated_log, LOG_FILE)

# --- 3. REINFORCED CHARTING ENGINE ---
def show_momentum_chart(symbol):
    """Renders interactive Plotly charts with 5 timeframe tabs."""
    timeframes = {
        "1D": ("1d", "5m"),
        "1W": ("5d", "30m"),
        "1M": ("1mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1wk")
    }
    
    # Create tabs for each timeframe
    tabs = st.tabs(list(timeframes.keys()))
    
    for i, tab in enumerate(tabs):
        label = list(timeframes.keys())[i]
        period, interval = timeframes[label]
        
        with tab:
            try:
                # Fetching data - auto_adjust=True helps ensure consistent column names
                chart_raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
                
                if not chart_raw.empty:
                    # Flatten data to ensure it is a 1D array for Plotly
                    y_values = chart_raw['Close'].values.flatten()
                    x_values = chart_raw.index
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x_values, 
                        y=y_values, 
                        mode='lines',
                        line=dict(color='#00ff88', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 136, 0.1)',
                        name="Price"
                    ))
                    
                    fig.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white"),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.warning(f"No chart data available for {symbol} ({label}).")
            except Exception as e:
                st.error(f"Chart Error: {e}")

# --- 4. ANALYTICS & MARKET DATA ENGINE ---
def get_market_data(symbol):
    """Calculates Price, RSI, Trend, and 1-Year Historical Performance."""
    try:
        ticker = yf.Ticker(symbol)
        # Pull 1 year of data to calculate long-term performance
        history = ticker.history(period="1y") 
        
        if history.empty:
            return None
        
        # Current Stats
        current_price = history['Close'].iloc[-1]
        
        # 1-Year Performance (vs first day of the 1y period)
        price_start = history['Close'].iloc[0]
        performance_1y = ((current_price - price_start) / price_start) * 100
        
        # Trend (Price vs 200-Day Moving Average)
        sma_200 = history['Close'].rolling(window=200).mean().iloc[-1]
        trend_status = "Bullish" if current_price > sma_200 else "Bearish"
        
        # Relative Strength Index (RSI - 14 Days)
        delta = history['Close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        rsi_val = 100 - (100 / (1 + rs)).iloc[-1]
        
        return {
            "name": ticker.info.get('longName', symbol),
            "price": current_price,
            "rsi": rsi_val,
            "trend": trend_status,
            "perf_1y": performance_1y,
            "news": ticker.news
        }
    except Exception:
        return None

def get_ai_analysis(symbol, name, tech_data, is_newbie=False):
    """Generates an AI recommendation using GPT-4o."""
    news_items = tech_data.get('news', [])
    headlines = [n.get('title', '') for n in news_items[:5]]
    news_context = "\n".join(headlines) if headlines else "No recent headlines found."
    
    style = "simplified beginner-friendly" if is_newbie else "expert analyst"
    
    prompt = f"""
    Investment Audit for {name} ({symbol}).
    User Level: {style}.
    Data: Price ${tech_data['price']:.2f}, RSI {tech_data['rsi']:.2f}, Trend {tech_data['trend']}, 1Y Change {tech_data['perf_1y']:.2f}%.
    Recent Headlines: {news_context}
    
    Format your response as: **[Score/10]** | **[REC]** | [One-sentence insight]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "Professional Investment Advisor"}, {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception:
        return "AI analysis failed. Please check your API key."

# --- 5. UI INITIALIZATION ---
st.set_page_config(page_title="StockWise Pro", page_icon="📈", layout="wide")

# Keeping the portfolio in session state for fast UI changes
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data(PORTFOLIO_FILE, ["Symbol", "Quantity", "Purchase_Price"])

# --- 6. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("StockWise AI Pro")
    app_mode = st.selectbox("Navigate Menu", ["Dashboard", "Mock Simulator", "Analyst Good Vibes", "Performance Log", "Management"])
    st.divider()
    st.markdown("""
    **Advisor Rules:**
    - **RSI > 70:** Overbought (Potential Peak)
    - **RSI < 30:** Oversold (Potential Opportunity)
    - **Bullish Trend:** Price is above its 200-day average.
    """)
    st.caption("Version 4.3 | Fortress Edition")

# --- 7. DASHBOARD (REAL HOLDINGS) ---
if app_mode == "Dashboard":
    st.title("Real Portfolio Dashboard 📊")
    
    if st.session_state.portfolio.empty:
        st.info("Your portfolio is currently empty. Add stocks in the 'Management' tab.")
    else:
        with st.spinner('Syncing with Global Markets & 1Y History...'):
            # Working copy for calculations
            work_df = st.session_state.portfolio.copy()
            
            # Temporary storage for calculation results
            names, prices, y1_perfs, metrics_store = [], [], [], []
            
            for sym in work_df['Symbol']:
                data = get_market_data(sym)
                if data:
                    names.append(data['name'])
                    prices.append(data['price'])
                    y1_perfs.append(data['perf_1y'])
                    metrics_store.append(data)
                else:
                    names.append("N/A")
                    prices.append(0.0)
                    y1_perfs.append(0.0)
                    metrics_store.append(None)
            
            # Map values back to the dataframe
            work_df['Name'] = names
            work_df['Live Price'] = prices
            work_df['1Y Perf %'] = y1_perfs
            work_df['Total Value'] = work_df['Quantity'] * work_df['Live Price']
            work_df['Total Gain'] = work_df['Total Value'] - (work_df['Quantity'] * work_df['Purchase_Price'])

        # Top-Level Summary Metrics
        m_col1, m_col2 = st.columns(2)
        grand_total_val = work_df['Total Value'].sum()
        grand_total_gain = work_df['Total Gain'].sum()
        
        m_col1.metric("Grand Portfolio Value", f"${grand_total_val:,.2f}")
        m_col2.metric(
            "Portfolio Profit/Loss", 
            f"${grand_total_gain:,.2f}", 
            f"{((grand_total_gain / (grand_total_val - grand_total_gain)) * 100):.2f}%" if (grand_total_val - grand_total_gain) != 0 else "0%"
        )

        # PERFORMANCE SUMMARY TABLE (Restored)
        st.subheader("Performance Summary Table")
        st.dataframe(
            work_df[['Symbol', 'Name', 'Quantity', 'Live Price', '1Y Perf %', 'Total Gain']].style.format(precision=2), 
            use_container_width=True
        )

        st.divider()
        
        # INDIVIDUAL DEEP DIVES
        st.subheader("Detailed Stock Analysis")
        for idx, row in work_df.iterrows():
            with st.expander(f"🔍 Deep Dive: {row['Name']} ({row['Symbol']})"):
                # Momentum Charts
                show_momentum_chart(row['Symbol'])
                
                # Analysis Logic
                if st.button(f"Analyze {row['Symbol']} (Auto-Log)", key=f"dash_audit_{row['Symbol']}"):
                    analysis_result = get_ai_analysis(row['Symbol'], row['Name'], metrics_store[idx])
                    st.info(analysis_result)
                    # Automatically log the analysis to history
                    log_recommendation(row['Symbol'], row['Name'], analysis_result, row['Live Price'])
                    st.toast(f"Analysis for {row['Symbol']} saved to log.")

# --- 8. MOCK SIMULATOR ---
elif app_mode == "Mock Simulator":
    st.title("Mock Investment Simulator 🧪")
    st.write("Visually track potential entry points and historical trends.")
    
    mock_data = load_data(MOCK_FILE, ["Symbol", "Quantity", "Watch_Price", "Date_Added"])
    
    with st.expander("➕ Add New Mock Stock"):
        with st.form("new_mock_stock"):
            ms = st.text_input("Ticker Symbol").upper().strip()
            mq = st.number_input("Virtual Quantity", min_value=0.1, value=10.0)
            mw = st.number_input("Virtual Entry Price", min_value=0.01)
            if st.form_submit_button("Start Simulation"):
                new_v_entry = pd.DataFrame([{
                    "Symbol": ms, "Quantity": mq, "Watch_Price": mw, "Date_Added": datetime.now().strftime("%Y-%m-%d")
                }])
                save_data(pd.concat([mock_data, new_v_entry], ignore_index=True), MOCK_FILE)
                st.rerun()

    if not mock_data.empty:
        for m_idx, m_row in mock_data.iterrows():
            market_stats = get_market_data(m_row['Symbol'])
            if market_stats:
                v_profit = (market_stats['price'] - m_row['Watch_Price']) * m_row['Quantity']
                with st.container(border=True):
                    st.subheader(f"{market_stats['name']} ({m_row['Symbol']})")
                    st.write(f"**1-Year Historical Trend:** {market_stats['perf_1y']:+.2f}%")
                    
                    # Show charts
                    show_momentum_chart(m_row['Symbol'])
                    
                    s_c1, s_c2, s_c3 = st.columns([2, 2, 1])
                    p_color = "green" if v_profit >= 0 else "red"
                    s_c1.markdown(f"**Potential Gain:** :{p_color}[${v_profit:,.2f}]")
                    s_c2.caption(f"Virtual Entry: ${m_row['Watch_Price']:.2f} | Current: ${market_stats['price']:.2f}")
                    
                    if s_c3.button("💎 Buy for Real", key=f"mock_buy_{m_idx}"):
                        st.session_state[f"flow_buy_{m_idx}"] = True
                
                if st.session_state.get(f"flow_buy_{m_idx}"):
                    with st.form(f"convert_mock_{m_idx}"):
                        st.write(f"### Finalize Real Purchase for {m_row['Symbol']}")
                        final_qty = st.number_input("Actual Quantity Bought", value=m_row['Quantity'])
                        final_prc = st.number_input("Final Execution Price", value=market_stats['price'])
                        
                        f_btn1, f_btn2 = st.columns(2)
                        if f_btn1.form_submit_button("Confirm & Move"):
                            # Move to Real
                            new_real_asset = pd.DataFrame([{"Symbol": m_row['Symbol'], "Quantity": final_qty, "Purchase_Price": final_prc}])
                            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_real_asset], ignore_index=True)
                            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                            # Remove from Mock
                            save_data(mock_data.drop(m_idx), MOCK_FILE)
                            del st.session_state[f"flow_buy_{m_idx}"]
                            st.rerun()
                        if f_btn2.form_submit_button("Cancel"):
                            del st.session_state[f"flow_buy_{m_idx}"]; st.rerun()
    else:
        st.info("No active simulations found.")

# --- 9. ANALYST GOOD VIBES ---
elif app_mode == "Analyst Good Vibes":
    st.title("Analyst 'Good Vibes' Picks 🌟")
    st.write("Top-tier consensus picks for 2026 with strong historical momentum.")
    
    good_vibes_list = {
        "MU": "Micron (Leading the AI-memory revolution)",
        "GOOG": "Alphabet (Diversified AI giant with high growth potential)",
        "ASML": "ASML (The backbone of semiconductor manufacturing)",
        "CHWY": "Chewy (Consumer growth and pet health expansion)"
    }
    
    for v_sym, v_desc in good_vibes_list.items():
        with st.expander(f"🌟 {v_sym} - {v_desc}"):
            v_market = get_market_data(v_sym)
            if v_market:
                st.write(f"**1-Year Performance:** {v_market['perf_1y']:+.2f}%")
            
            show_momentum_chart(v_sym)
            
            if st.button(f"Analyze {v_sym}", key=f"vibe_btn_{v_sym}"):
                v_analysis = get_ai_analysis(v_sym, v_sym, v_market, is_newbie=True)
                st.success(v_analysis)
                # Auto-log these as well
                log_recommendation(v_sym, v_sym, v_analysis, v_market['price'])

# --- 10. PERFORMANCE LOG ---
elif app_mode == "Performance Log":
    st.title("Historical Advisor Log 📈")
    history_log = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    
    if history_log.empty:
        st.info("No logs found. Run analyses in the Dashboard or Good Vibes sections.")
    else:
        st.write("Review past AI performance and entry price predictions:")
        st.dataframe(history_log.sort_values(by="Date", ascending=False), use_container_width=True)

# --- 11. MANAGEMENT ---
elif app_mode == "Management":
    st.title("Portfolio Controls ⚙️")
    
    with st.form("manual_add_asset"):
        st.subheader("Add Real Investment")
        man_cols = st.columns(3)
        man_s = man_cols[0].text_input("Ticker").upper().strip()
        man_q = man_cols[1].number_input("Quantity", min_value=0.0)
        man_p = man_cols[2].number_input("Purchase Price", min_value=0.0)
        
        if st.form_submit_button("Save to Dashboard"):
            if man_s:
                new_h = pd.DataFrame([{"Symbol": man_s, "Quantity": man_q, "Purchase_Price": man_p}])
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_h], ignore_index=True)
                save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                st.success(f"Added {man_s} to records.")
                st.rerun()

    if not st.session_state.portfolio.empty:
        st.divider()
        st.subheader("Delete Record")
        to_purge = st.selectbox("Ticker to remove permanently:", st.session_state.portfolio['Symbol'])
        if st.button("🗑️ Delete Permanently"):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != to_purge]
            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
            st.warning(f"Purged {to_purge} from the records.")
            st.rerun()

# --- FOOTER ---
st.sidebar.divider()
st.sidebar.caption(f"System Sync: {datetime.now().strftime('%H:%M:%S')}")