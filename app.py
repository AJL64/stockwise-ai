import streamlit as st
import pandas as pd
import os
import yfinance as yf
from openai import OpenAI
import requests
from datetime import datetime
import plotly.graph_objects as go

# --- CONFIGURATION & SECRETS ---
# These are pulled from your Streamlit Cloud Secrets dashboard
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)
PORTFOLIO_FILE = "my_portfolio.csv"
MOCK_FILE = "mock_portfolio.csv"
LOG_FILE = "advisor_log.csv"

# --- DATA STORAGE ENGINE ---
def load_data(file, columns):
    if os.path.exists(file): 
        try:
            return pd.read_csv(file)
        except:
            return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def save_data(df, file):
    df.to_csv(file, index=False)

def log_recommendation(symbol, name, analysis, price):
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
    """Generates interactive charts for 1D, 1W, 1M, 6M, and 1Y timeframes."""
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
                # Use progress=False to keep the UI clean
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
                        height=250,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color="white")
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.warning("No data found for this timeframe.")
            except Exception as e:
                st.error(f"Chart Error: {e}")

# --- ANALYTICS ENGINE ---
def get_tech_indicators(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="250d")
        if hist.empty: return None
        
        info = ticker.info
        current_price = hist['Close'].iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # RSI Calculation
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
    news_items = tech_data.get('news', [])
    titles = [item.get('title') or "" for item in news_items[:5]]
    combined_news = "\n".join(filter(None, titles)) if titles else "No recent headlines."
    
    tone = "beginner-friendly with 'Good Vibes' focus" if newbie_mode else "professional and detailed"
    
    prompt = f"""
    Analyze {name} ({symbol}). Style: {tone}.
    Data: Price ${tech_data['price']:.2f}, RSI {tech_data['rsi']:.2f}, Trend {tech_data['trend']}.
    Recent Headlines: {combined_news}
    Return exactly: **[Score/10]** | **[REC]** | [One-sentence insight]
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

# --- UI SETUP ---
st.set_page_config(page_title="StockWise Pro", page_icon="📈", layout="wide")

# Persistent State for Smooth Navigation
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data(PORTFOLIO_FILE, ["Symbol", "Quantity", "Purchase_Price"])

# --- SIDEBAR ---
with st.sidebar:
    st.title("StockWise AI Pro")
    app_mode = st.selectbox("Navigate", ["Dashboard", "Mock Simulator", "Analyst Good Vibes", "Performance Log", "Management"])
    st.divider()
    st.markdown("""
    **Indicators:**
    - **RSI > 70:** Overbought (Hot)
    - **RSI < 30:** Oversold (Opportunity)
    - **Bullish:** Price is above 200-day average.
    """)

# --- DASHBOARD (REAL HOLDINGS) ---
if app_mode == "Dashboard":
    st.title("Real Portfolio Dashboard 📊")
    if st.session_state.portfolio.empty:
        st.info("No holdings found. Go to 'Management' to add your first stock.")
    else:
        with st.spinner('Syncing with Wall Street...'):
            df = st.session_state.portfolio.copy()
            names, prices, rsis, trends, metrics = [], [], [], [], []
            for sym in df['Symbol']:
                data = get_tech_indicators(sym)
                if data:
                    names.append(data['name']); prices.append(data['price'])
                    rsis.append(data['rsi']); trends.append(data['trend']); metrics.append(data)
                else:
                    names.append("N/A"); prices.append(0); rsis.append(0); trends.append("N/A"); metrics.append(None)
            
            df['Name'], df['Price'], df['RSI'], df['Trend'] = names, prices, rsis, trends
            df['Value'] = df['Quantity'] * df['Price']
            df['Gain'] = df['Value'] - (df['Quantity'] * df['Purchase_Price'])

        c1, c2 = st.columns(2)
        c1.metric("Total Portfolio Value", f"${df['Value'].sum():,.2f}")
        c2.metric("Total Gain/Loss", f"${df['Gain'].sum():,.2f}", f"{((df['Gain'].sum() / (df['Value'].sum() - df['Gain'].sum())) * 100):.2f}%" if (df['Value'].sum() - df['Gain'].sum()) != 0 else "0%")

        st.divider()
        for i, row in df.iterrows():
            with st.expander(f"{row['Name']} ({row['Symbol']}) | Profit: ${row['Gain']:,.2f}"):
                show_momentum_chart(row['Symbol'])
                if st.button(f"Analyze {row['Symbol']}", key=f"an_{row['Symbol']}"):
                    analysis = get_ai_analysis(row['Symbol'], row['Name'], metrics[i])
                    st.write(analysis)
                    if st.button(f"Log this Audit", key=f"log_{row['Symbol']}"):
                        log_recommendation(row['Symbol'], row['Name'], analysis, row['Price'])
                        st.toast("Saved to Performance Log!")

# --- MOCK SIMULATOR ---
elif app_mode == "Mock Simulator":
    st.title("Mock Investment Simulator 🧪")
    st.write("Track potential entries visually before putting real money on the line.")
    
    mock_df = load_data(MOCK_FILE, ["Symbol", "Quantity", "Watch_Price", "Date_Added"])
    
    with st.expander("➕ Add New Simulation"):
        with st.form("new_mock"):
            ms = st.text_input("Ticker Symbol (e.g. NVDA)").upper().strip()
            mq = st.number_input("Virtual Quantity", value=10.0)
            mw = st.number_input("Virtual Entry Price", value=0.0)
            if st.form_submit_button("Start Simulation"):
                new_row = pd.DataFrame([{"Symbol": ms, "Quantity": mq, "Watch_Price": mw, "Date_Added": datetime.now().strftime("%Y-%m-%d")}])
                save_data(pd.concat([mock_df, new_row], ignore_index=True), MOCK_FILE)
                st.rerun()

    if not mock_df.empty:
        for idx, row in mock_df.iterrows():
            data = get_tech_indicators(row['Symbol'])
            if data:
                gain = (data['price'] - row['Watch_Price']) * row['Quantity']
                with st.container(border=True):
                    st.subheader(f"{data['name']} ({row['Symbol']})")
                    show_momentum_chart(row['Symbol'])
                    
                    col_m1, col_m2, col_m3 = st.columns([2, 2, 1])
                    color = "green" if gain >= 0 else "red"
                    col_m1.markdown(f"**Virtual Profit:** :{'green' if gain >= 0 else 'red'}[${gain:,.2f}]")
                    col_m2.caption(f"Simulating since {row['Date_Added']} at ${row['Watch_Price']:.2f}")
                    
                    if col_m3.button("💎 Buy for Real", key=f"buy_{idx}"):
                        st.session_state[f"active_buy_{idx}"] = True
                
                if st.session_state.get(f"active_buy_{idx}"):
                    with st.form(f"finalize_{idx}"):
                        st.write("### Finalize Real Purchase")
                        final_q = st.number_input("Real Quantity", value=row['Quantity'])
                        final_p = st.number_input("Real Price Paid", value=data['price'])
                        ca, cb = st.columns(2)
                        if ca.form_submit_button("Move to Portfolio"):
                            # Add to Real
                            new_real = pd.DataFrame([{"Symbol": row['Symbol'], "Quantity": final_q, "Purchase_Price": final_p}])
                            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_real], ignore_index=True)
                            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                            # Remove from Mock
                            save_data(mock_df.drop(idx), MOCK_FILE)
                            del st.session_state[f"active_buy_{idx}"]
                            st.rerun()
                        if cb.form_submit_button("Cancel"):
                            del st.session_state[f"active_buy_{idx}"]
                            st.rerun()
    else:
        st.info("No active simulations. Add one to see the chart momentum.")

# --- ANALYST GOOD VIBES ---
elif app_mode == "Analyst Good Vibes":
    st.title("Top Analyst Consensus (May 2026) 🌟")
    st.write("Fresh opportunities with strong technical momentum and high conviction.")
    
    # 2026 High Conviction List
    picks = {
        "MU": "Micron (AI Infrastructure Backbone)",
        "GOOG": "Alphabet (Undervalued Tech Giant)",
        "ASML": "ASML (Essential Chip Equipment)",
        "CHWY": "Chewy (Consumer Sentiment Play)"
    }
    
    for sym, desc in picks.items():
        with st.expander(f"{sym} - {desc}"):
            show_momentum_chart(sym)
            if st.button(f"Get Advisor Verdict on {sym}"):
                data = get_tech_indicators(sym)
                if data:
                    analysis = get_ai_analysis(sym, sym, data, newbie_mode=True)
                    st.success(analysis)
                    log_recommendation(sym, sym, analysis, data['price'])

# --- PERFORMANCE LOG ---
elif app_mode == "Performance Log":
    st.title("Advisor Track Record 📈")
    log_df = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    if log_df.empty:
        st.info("No audit logs yet.")
    else:
        st.dataframe(log_df.sort_values(by="Date", ascending=False), use_container_width=True)

# --- MANAGEMENT ---
elif app_mode == "Management":
    st.title("Management & Settings ⚙️")
    with st.form("manual_add"):
        st.subheader("Add Real Stock Entry")
        col_add = st.columns(3)
        msy = col_add[0].text_input("Ticker").upper()
        mqt = col_add[1].number_input("Quantity", min_value=0.0)
        mpr = col_add[2].number_input("Purchase Price", min_value=0.0)
        if st.form_submit_button("Save to Dashboard"):
            if msy:
                new_row = pd.DataFrame([{"Symbol": msy, "Quantity": mqt, "Purchase_Price": mpr}])
                st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                st.success(f"Added {msy}")
                st.rerun()

    if not st.session_state.portfolio.empty:
        st.divider()
        st.subheader("Remove Stock")
        to_del = st.selectbox("Select ticker to remove:", st.session_state.portfolio['Symbol'])
        if st.button("🗑️ Delete Permanently"):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != to_del]
            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
            st.rerun()

st.sidebar.divider()
st.sidebar.caption("v3.5 | Final Advisor Pro")