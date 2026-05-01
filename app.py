import streamlit as st
import pandas as pd
import os
import yfinance as yf
from openai import OpenAI
import requests
from datetime import datetime

# --- CONFIGURATION & SECRETS ---
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)
PORTFOLIO_FILE = "my_portfolio.csv"
LOG_FILE = "advisor_log.csv"

# --- DATA HANDLING ---
def load_data(file, columns):
    if os.path.exists(file): return pd.read_csv(file)
    return pd.DataFrame(columns=columns)

def save_data(df, file): df.to_csv(file, index=False)

def log_recommendation(symbol, name, analysis, price):
    log_df = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    new_row = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Symbol": symbol,
        "Name": name,
        "Price_At_Rec": round(price, 2),
        "Analysis": analysis
    }])
    log_df = pd.concat([log_df, new_row], ignore_index=True)
    save_data(log_df, LOG_FILE)

# --- TECHNICAL CALCULATIONS ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_tech_indicators(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="250d")
        if hist.empty: return None
        info = ticker.info
        current_price = hist['Close'].iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        return {
            "name": info.get('longName', symbol),
            "price": current_price,
            "rsi": rsi,
            "trend": "Bullish" if current_price > sma_200 else "Bearish",
            "news": ticker.news
        }
    except: return None

# --- AI & NOTIFICATIONS ---
def send_telegram_msg(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try: requests.post(url, json=payload)
        except: pass

def get_ai_analysis(symbol, name, tech_data, newbie_mode=False):
    news_items = tech_data.get('news', [])
    titles = [item.get('title') or "" for item in news_items[:7]]
    combined_news = "\n".join(filter(None, titles)) if titles else "No news."
    
    extra_context = "Special Request: Analyze this for a 'Newbie' looking for 'Good Vibes' and high success probability." if newbie_mode else ""

    prompt = f"""
    Expert Analysis for {name} ({symbol}):
    {extra_context}
    Technicals: RSI {tech_data['rsi']:.2f}, Trend {tech_data['trend']}.
    News: {combined_news}
    Format: **[Score/10]** | **[REC]** | [One-sentence insight]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional investment advisor."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except: return "Analysis Error"

# --- UI SETUP ---
st.set_page_config(page_title="StockWise Advisor Pro", page_icon="🌟", layout="wide")

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data(PORTFOLIO_FILE, ["Symbol", "Quantity", "Purchase_Price"])

# --- SIDEBAR ---
with st.sidebar:
    st.title("StockWise Advisor")
    app_mode = st.selectbox("Navigation", ["Dashboard", "Analyst Good Vibes", "Performance Log", "Portfolio Management"])
    st.divider()
    st.markdown("**Legend:**\n* RSI > 70: Hot/Overbought\n* RSI < 30: Cheap/Oversold\n* Bullish: Safe long-term trend")

# --- DASHBOARD ---
if app_mode == "Dashboard":
    st.title("My Portfolio 📊")
    if st.session_state.portfolio.empty:
        st.info("Your portfolio is empty. Go to Management to add stocks.")
    else:
        with st.spinner('Updating Market Data...'):
            df = st.session_state.portfolio.copy()
            names, prices, rsis, trends, metrics = [], [], [], [], []
            for sym in df['Symbol']:
                data = get_tech_indicators(sym)
                if data:
                    names.append(data['name']); prices.append(data['price']); rsis.append(data['rsi'])
                    trends.append(data['trend']); metrics.append(data)
                else:
                    names.append("N/A"); prices.append(0); rsis.append(0); trends.append("N/A"); metrics.append(None)
            
            df['Name'], df['Price'], df['RSI'], df['Trend'] = names, prices, rsis, trends
            df['Value'] = df['Quantity'] * df['Price']
            df['Gain'] = df['Value'] - (df['Quantity'] * df['Purchase_Price'])

        c1, c2 = st.columns(2)
        c1.metric("Total Value", f"${df['Value'].sum():,.2f}")
        c2.metric("Total Gain/Loss", f"${df['Gain'].sum():,.2f}")
        st.dataframe(df[['Symbol', 'Name', 'Quantity', 'Price', 'RSI', 'Trend', 'Gain']].style.format(precision=2), use_container_width=True)
        
        if st.button("Run AI Audit"):
            for i, sym in enumerate(df['Symbol']):
                if metrics[i]:
                    with st.expander(f"Audit: {df['Name'].iloc[i]}"):
                        analysis = get_ai_analysis(sym, df['Name'].iloc[i], metrics[i])
                        st.write(analysis)
                        if st.button(f"Log this {sym} Audit", key=f"log_{sym}"):
                            log_recommendation(sym, df['Name'].iloc[i], analysis, df['Price'].iloc[i])
                            st.toast(f"Logged {sym}!")

# --- ANALYST GOOD VIBES ---
elif app_mode == "Analyst Good Vibes":
    st.title("Newbie Friendly & Strong Potential 🌟")
    st.write("These companies have high analyst scores and 'Good Vibes' for 2026.")
    
    vibes_list = {
        "MU": "Micron (AI Memory Backbone)",
        "SG": "Sweetgreen (High-Growth Retail)",
        "HRMY": "Harmony Bio (Strong Medical Pipeline)",
        "AMPL": "Amplitude (Digital Intelligence)",
        "CWCO": "Consolidated Water (Safe Infrastructure)"
    }
    
    for sym, desc in vibes_list.items():
        with st.expander(f"{sym} - {desc}"):
            if st.button(f"Analyze {sym} Opportunity"):
                data = get_tech_indicators(sym)
                if data:
                    analysis = get_ai_analysis(sym, data['name'], data, newbie_mode=True)
                    st.success(analysis)
                    log_recommendation(sym, data['name'], analysis, data['price'])
                    st.info("Successfully added to Performance Log.")

# --- PERFORMANCE LOG ---
elif app_mode == "Performance Log":
    st.title("Historical AI Advisor Track Record 📈")
    log_df = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    if log_df.empty:
        st.info("No recommendations have been logged yet.")
    else:
        st.write("Here is what the AI predicted in the past:")
        st.dataframe(log_df.sort_values(by="Date", ascending=False), use_container_width=True)

# --- PORTFOLIO MANAGEMENT ---
elif app_mode == "Portfolio Management":
    st.title("Manage Holdings ⚙️")
    with st.form("add_stock"):
        c = st.columns(3)
        s = c[0].text_input("Ticker").upper().strip()
        q = c[1].number_input("Quantity", min_value=0.0)
        p = c[2].number_input("Purchase Price", min_value=0.0)
        if st.form_submit_button("Save"):
            if s:
                new_df = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != s]
                new_row = pd.DataFrame([{"Symbol": s, "Quantity": q, "Purchase_Price": p}])
                st.session_state.portfolio = pd.concat([new_df, new_row], ignore_index=True)
                save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                st.rerun()

    if not st.session_state.portfolio.empty:
        st.divider()
        to_del = st.selectbox("Remove Stock:", st.session_state.portfolio['Symbol'])
        if st.button("🗑️ Delete Selected"):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != to_del]
            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
            st.rerun()

st.sidebar.divider()
st.sidebar.caption("v3.0 | Advisor Pro & Performance Tracking")