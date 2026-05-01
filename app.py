import streamlit as st
import pandas as pd
import os
import yfinance as yf
from openai import OpenAI
import requests
from datetime import datetime
import plotly.graph_objects as go

# --- CONFIGURATION & SECRETS ---
# Pulling credentials from Streamlit's secure vault
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)

# File paths for persistent storage
PORTFOLIO_FILE = "my_portfolio.csv"
MOCK_FILE = "mock_portfolio.csv"
LOG_FILE = "advisor_log.csv"

# --- DATA STORAGE ENGINE ---
def load_data(file, columns):
    """Loads CSV data or creates an empty DataFrame if file doesn't exist."""
    if os.path.exists(file): 
        try:
            return pd.read_csv(file)
        except:
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def save_data(df, file):
    """Saves the DataFrame to a CSV file."""
    df.to_csv(file, index=False)

def log_recommendation(symbol, name, analysis, price):
    """Saves AI analyst recommendations to the historical log."""
    log_df = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    new_row = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Symbol": symbol,
        "Name": name,
        "Price_At_Rec": round(price, 2),
        "Analysis": analysis
    }])
    updated_log = pd.concat([log_df, new_row], ignore_index=True)
    save_data(updated_log, LOG_FILE)

# --- VISUAL CHARTING ENGINE ---
def show_momentum_chart(symbol):
    """Generates interactive Plotly charts for multiple timeframes."""
    intervals = {
        "1D": ("1d", "5m"),
        "1W": ("5d", "30m"),
        "1M": ("1mo", "1d"),
        "6M": ("6mo", "1d"),
        "1Y": ("1y", "1wk")
    }
    
    tabs = st.tabs(list(intervals.keys()))
    
    for i, tab in enumerate(tabs):
        label = list(intervals.keys())[i]
        period, interval = intervals[label]
        
        with tab:
            try:
                # Fetching historical data from Yahoo Finance
                data = yf.download(symbol, period=period, interval=interval, progress=False)
                if not data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index, 
                        y=data['Close'], 
                        mode='lines',
                        line=dict(color='#00ff88', width=2),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 136, 0.1)'
                    ))
                    fig.update_layout(
                        height=300,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.warning("No market data found for this timeframe.")
            except Exception as e:
                st.error(f"Chart Engine Error: {e}")

# --- ANALYTICS & AI ENGINE ---
def get_tech_indicators(symbol):
    """Fetches real-time price, RSI, and trend data."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="250d")
        if hist.empty: return None
        
        info = ticker.info
        current_price = hist['Close'].iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Relative Strength Index (RSI) Calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        return {
            "name": info.get('longName', symbol),
            "price": current_price,
            "rsi": rsi,
            "trend": "Bullish" if current_price > sma_200 else "Bearish",
            "news": ticker.news
        }
    except:
        return None

def get_ai_analysis(symbol, name, tech_data, newbie_mode=False):
    """Uses GPT-4o to analyze technicals and news for a specific stock."""
    news_items = tech_data.get('news', [])
    titles = [item.get('title') or "" for item in news_items[:5]]
    combined_news = "\n".join(filter(None, titles)) if titles else "No current news."
    
    tone = "beginner-friendly with 'Good Vibes' focus" if newbie_mode else "professional and data-driven"
    
    prompt = f"""
    Act as a professional stock analyst. Analyze {name} ({symbol}).
    Mode: {tone}.
    Current Price: ${tech_data['price']:.2f}
    RSI: {tech_data['rsi']:.2f}
    Trend: {tech_data['trend']}
    Recent Headlines: {combined_news}
    
    Provide exactly: **[Score/10]** | **[REC]** | [One-sentence insight]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a top-tier investment advisor."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except:
        return "AI analysis currently unavailable."

# --- UI INITIALIZATION ---
st.set_page_config(page_title="StockWise Advisor Pro", page_icon="📈", layout="wide")

# Load session state for fast navigation
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data(PORTFOLIO_FILE, ["Symbol", "Quantity", "Purchase_Price"])

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("StockWise AI Pro")
    app_mode = st.selectbox("Navigation Menu", ["Dashboard", "Mock Simulator", "Analyst Good Vibes", "Performance Log", "Management"])
    st.divider()
    st.markdown("""
    **Financial Health Legend:**
    - **RSI > 70:** Overbought (Potential Peak)
    - **RSI < 30:** Oversold (Potential Entry)
    - **Bullish:** Price is trending above 200-day average.
    """)

# --- DASHBOARD (REAL HOLDINGS) ---
if app_mode == "Dashboard":
    st.title("Real Portfolio Dashboard 📊")
    if st.session_state.portfolio.empty:
        st.info("Your portfolio is currently empty. Visit the 'Management' tab to add stocks.")
    else:
        with st.spinner('Syncing data with market exchanges...'):
            df = st.session_state.portfolio.copy()
            names, prices, rsis, trends, metrics = [], [], [], [], []
            for sym in df['Symbol']:
                data = get_tech_indicators(sym)
                if data:
                    names.append(data['name'])
                    prices.append(data['price'])
                    rsis.append(data['rsi'])
                    trends.append(data['trend'])
                    metrics.append(data)
                else:
                    names.append("N/A")
                    prices.append(0)
                    rsis.append(0)
                    trends.append("N/A")
                    metrics.append(None)
            
            df['Name'], df['Price'], df['RSI'], df['Trend'] = names, prices, rsis, trends
            df['Value'] = df['Quantity'] * df['Price']
            df['Gain'] = df['Value'] - (df['Quantity'] * df['Purchase_Price'])

        # Top Level Metrics
        col_m1, col_m2 = st.columns(2)
        total_val = df['Value'].sum()
        total_gain = df['Gain'].sum()
        col_m1.metric("Total Portfolio Value", f"${total_val:,.2f}")
        col_m2.metric("Total Profit/Loss", f"${total_gain:,.2f}", f"{((total_gain / (total_val - total_gain)) * 100):.2f}%" if (total_val - total_gain) != 0 else "0%")

        # Performance Table
        st.subheader("Holdings Summary")
        st.dataframe(df[['Symbol', 'Name', 'Quantity', 'Price', 'RSI', 'Trend', 'Gain']].style.format(precision=2), use_container_width=True)

        st.divider()
        st.subheader("Deep Dive Analysis & Charts")
        for i, row in df.iterrows():
            with st.expander(f"🔍 {row['Name']} ({row['Symbol']}) | RSI: {row['RSI']:.1f}"):
                show_momentum_chart(row['Symbol'])
                if st.button(f"Analyze {row['Symbol']}", key=f"audit_{row['Symbol']}"):
                    audit_res = get_ai_analysis(row['Symbol'], row['Name'], metrics[i])
                    st.write(audit_res)
                    if st.button(f"Log Audit for {row['Symbol']}", key=f"log_btn_{row['Symbol']}"):
                        log_recommendation(row['Symbol'], row['Name'], audit_res, row['Price'])
                        st.toast(f"Recommendation for {row['Symbol']} saved!")

# --- MOCK SIMULATOR ---
elif app_mode == "Mock Simulator":
    st.title("Mock Investment Simulator 🧪")
    st.write("Stress-test your investment ideas without using real capital.")
    
    mock_df = load_data(MOCK_FILE, ["Symbol", "Quantity", "Watch_Price", "Date_Added"])
    
    with st.expander("➕ Create New Simulation"):
        with st.form("new_mock_form"):
            msy = st.text_input("Ticker Symbol (e.g., TSLA)").upper().strip()
            mqu = st.number_input("Virtual Quantity", min_value=0.1, value=10.0)
            mpr = st.number_input("Virtual Entry Price", min_value=0.01)
            if st.form_submit_button("Launch Simulation"):
                new_mock = pd.DataFrame([{"Symbol": msy, "Quantity": mqu, "Watch_Price": mpr, "Date_Added": datetime.now().strftime("%Y-%m-%d")}])
                updated_mock = pd.concat([mock_df, new_mock], ignore_index=True)
                save_data(updated_mock, MOCK_FILE)
                st.success(f"Simulation for {msy} is now live!")
                st.rerun()

    if not mock_df.empty:
        st.subheader("Active Virtual Holdings")
        for idx, row in mock_df.iterrows():
            m_data = get_tech_indicators(row['Symbol'])
            if m_data:
                m_gain = (m_data['price'] - row['Watch_Price']) * row['Quantity']
                with st.container(border=True):
                    st.markdown(f"### {m_data['name']} ({row['Symbol']})")
                    show_momentum_chart(row['Symbol'])
                    
                    c_sim1, c_sim2, c_sim3 = st.columns([2, 2, 1])
                    m_color = "green" if m_gain >= 0 else "red"
                    c_sim1.markdown(f"**Virtual Gain:** :{'green' if m_gain >= 0 else 'red'}[${m_gain:,.2f}]")
                    c_sim2.caption(f"Entry: ${row['Watch_Price']:.2f} | Current: ${m_data['price']:.2f}")
                    
                    if c_sim3.button("💎 Buy for Real", key=f"conv_real_{idx}"):
                        st.session_state[f"buying_now_{idx}"] = True
                
                if st.session_state.get(f"buying_now_{idx}"):
                    with st.form(f"final_conversion_{idx}"):
                        st.write(f"### Moving {row['Symbol']} to Real Portfolio")
                        fqty = st.number_input("Actual Quantity Purchased", value=row['Quantity'])
                        fprc = st.number_input("Actual Price Paid", value=m_data['price'])
                        
                        btn_c1, btn_c2 = st.columns(2)
                        if btn_c1.form_submit_button("Confirm & Move"):
                            # Add to Real Portfolio
                            new_real_entry = pd.DataFrame([{"Symbol": row['Symbol'], "Quantity": fqty, "Purchase_Price": fprc}])
                            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_real_entry], ignore_index=True)
                            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                            # Delete from Mock
                            save_data(mock_df.drop(idx), MOCK_FILE)
                            del st.session_state[f"buying_now_{idx}"]
                            st.success(f"Successfully converted {row['Symbol']} to a real asset!")
                            st.rerun()
                        if btn_c2.form_submit_button("Cancel"):
                            del st.session_state[f"buying_now_{idx}"]
                            st.rerun()
    else:
        st.info("No active simulations found.")

# --- ANALYST GOOD VIBES ---
elif app_mode == "Analyst Good Vibes":
    st.title("Analyst 'Good Vibes' Picks 🌟")
    st.write("Top conviction picks for 2026 based on analyst sentiment and technical momentum.")
    
    discovery_picks = {
        "MU": "Micron (Leading the AI memory transition)",
        "GOOG": "Alphabet (Massive scale and AI integration)",
        "ASML": "ASML (The gatekeeper of the semiconductor world)",
        "CHWY": "Chewy (Turning the corner on retail sentiment)"
    }
    
    for pick_sym, pick_desc in discovery_picks.items():
        with st.expander(f"🌟 {pick_sym} - {pick_desc}"):
            show_momentum_chart(pick_sym)
            if st.button(f"Advisor Audit: {pick_sym}", key=f"vibe_audit_{pick_sym}"):
                p_data = get_tech_indicators(pick_sym)
                if p_data:
                    vibe_res = get_ai_analysis(pick_sym, p_data['name'], p_data, newbie_mode=True)
                    st.success(vibe_res)
                    log_recommendation(pick_sym, p_data['name'], vibe_res, p_data['price'])

# --- PERFORMANCE LOG ---
elif app_mode == "Performance Log":
    st.title("Historical Advisor Track Record 📈")
    hist_df = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    if hist_df.empty:
        st.info("No audit logs have been recorded yet.")
    else:
        st.write("Review past AI recommendations and entry prices:")
        st.dataframe(hist_df.sort_values(by="Date", ascending=False), use_container_width=True)

# --- MANAGEMENT ---
elif app_mode == "Management":
    st.title("Portfolio Settings & Controls ⚙️")
    
    with st.form("manual_stock_addition"):
        st.subheader("Add Real Investment Entry")
        col_in1, col_in2, col_in3 = st.columns(3)
        add_sym = col_in1.text_input("Ticker Symbol").upper().strip()
        add_qty = col_in2.number_input("Shares Quantity", min_value=0.0)
        add_prc = col_in3.number_input("Average Purchase Price", min_value=0.0)
        
        if st.form_submit_button("Add to Real Portfolio"):
            if add_sym:
                new_hld = pd.DataFrame([{"Symbol": add_sym, "Quantity": add_qty, "Purchase_Price": add_prc}])
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_hld], ignore_index=True)
                save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                st.success(f"Added {add_sym} to holdings.")
                st.rerun()

    if not st.session_state.portfolio.empty:
        st.divider()
        st.subheader("Liquidate/Remove Asset")
        target_del = st.selectbox("Select ticker to remove permanently:", st.session_state.portfolio['Symbol'])
        if st.button("🗑️ Delete Selected Holding"):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != target_del]
            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
            st.warning(f"Removed {target_del} from your records.")
            st.rerun()

# --- FOOTER ---
st.sidebar.divider()
st.sidebar.caption(f"v3.7 | Advisor Pro | Last Sync: {datetime.now().strftime('%H:%M:%S')}")