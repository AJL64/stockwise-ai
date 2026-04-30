# --- Configuration & API Keys ---
import streamlit as st
import pandas as pd
import os
import yfinance as yf
from openai import OpenAI
import requests

# --- CONFIGURATION & API KEYS ---
# --- CONFIGURATION & SECRETS ---
# Streamlit Cloud will now pull these from the "Secrets" menu automatically
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
TELEGRAM_TOKEN = st.secrets["TELEGRAM_TOKEN"]
TELEGRAM_CHAT_ID = st.secrets["TELEGRAM_CHAT_ID"]

client = OpenAI(api_key=OPENAI_API_KEY)
PORTFOLIO_FILE = "my_portfolio.csv"

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

# --- CORE FUNCTIONS ---
def send_telegram_msg(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
        try: requests.post(url, json=payload)
        except: pass

def load_data():
    if os.path.exists(PORTFOLIO_FILE): return pd.read_csv(PORTFOLIO_FILE)
    return pd.DataFrame(columns=["Symbol", "Quantity", "Purchase_Price"])

def save_data(df): df.to_csv(PORTFOLIO_FILE, index=False)

def get_ai_analysis(symbol, name, tech_data):
    news_items = tech_data.get('news', [])
    titles = [item.get('title') or item.get('headline') or "" for item in news_items[:7]]
    combined_news = "\n".join(filter(None, titles)) if titles else "No news."
    
    rsi = tech_data['rsi']
    prompt = f"""
    Analyze {name} ({symbol}):
    - Tech: RSI {rsi:.2f}, Trend {tech_data['trend']}.
    - News: {combined_news}
    Format: **[Score/10]** | **[REC]** | [Insight]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional analyst."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except: return "Analysis Error"

# --- UI SETUP ---
st.set_page_config(page_title="StockWise AI Pro", page_icon="📈", layout="wide")

# Sidebar Legend
with st.sidebar:
    st.title("StockWise AI Pro")
    app_mode = st.selectbox("Navigation", ["Dashboard", "Market Opportunities", "Portfolio Management"])
    st.divider()
    st.header("💡 Legend")
    st.markdown("""
    **RSI (Momentum)**
    * **> 70**: Overbought (High)
    * **< 30**: Oversold (Low)
    **Trend (200-day SMA)**
    * **Bullish**: Price > Average.
    * **Bearish**: Price < Average.
    """)

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data()

# --- DASHBOARD ---
if app_mode == "Dashboard":
    st.title("My Portfolio 📊")
    if st.session_state.portfolio.empty:
        st.info("Your portfolio is empty. Add stocks in 'Portfolio Management'.")
    else:
        with st.spinner('Updating Market Data...'):
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
                    names.append("N/A"); prices.append(0); rsis.append(0); trends.append("N/A"); metrics.append(None)
            
            df['Name'] = names
            df['Price'] = prices
            df['RSI'] = rsis
            df['Trend'] = trends
            df['Value'] = df['Quantity'] * df['Price']
            df['Gain'] = df['Value'] - (df['Quantity'] * df['Purchase_Price'])

        m1, m2 = st.columns(2)
        m1.metric("Total Value", f"${df['Value'].sum():,.2f}")
        m2.metric("Total Gain/Loss", f"${df['Gain'].sum():,.2f}")
        
        st.dataframe(df[['Symbol', 'Name', 'Quantity', 'Price', 'RSI', 'Trend', 'Gain']].style.format(precision=2), use_container_width=True)
        
        if st.button("Run AI Audit"):
            for i, sym in enumerate(df['Symbol']):
                if metrics[i]:
                    with st.expander(f"Audit: {df['Name'].iloc[i]} ({sym})"):
                        st.write(get_ai_analysis(sym, df['Name'].iloc[i], metrics[i]))

# --- MARKET OPPORTUNITIES ---
elif app_mode == "Market Opportunities":
    st.title("Discovery Engine 🔍")
    scan_list = ["NVDA", "TSLA", "AAPL", "MSFT", "AMD", "META", "AMZN", "GOOGL", "AVGO", "SMCI"]
    
    if st.button("Scan Market"):
        progress = st.progress(0)
        for i, sym in enumerate(scan_list):
            data = get_tech_indicators(sym)
            if data:
                st.subheader(f"{data['name']} ({sym})")
                analysis = get_ai_analysis(sym, data['name'], data)
                if "BUY" in analysis: st.success(analysis)
                elif "SELL" in analysis: st.error(analysis)
                else: st.warning(analysis)
            progress.progress((i + 1) / len(scan_list))
        st.balloons()

# --- PORTFOLIO MANAGEMENT ---
elif app_mode == "Portfolio Management":
    st.title("Manage Holdings ⚙️")
    
    st.subheader("Add / Update Stock")
    with st.form("add_stock"):
        c = st.columns(3)
        s = c[0].text_input("Ticker Symbol").upper().strip()
        q = c[1].number_input("Total Quantity", min_value=0.0)
        p = c[2].number_input("Purchase Price", min_value=0.0)
        if st.form_submit_button("Save to Portfolio"):
            if s:
                new_df = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != s]
                new_row = pd.DataFrame([{"Symbol": s, "Quantity": q, "Purchase_Price": p}])
                st.session_state.portfolio = pd.concat([new_df, new_row], ignore_index=True)
                save_data(st.session_state.portfolio)
                st.success(f"Saved {s}")
                st.rerun()

    if not st.session_state.portfolio.empty:
        st.divider()
        st.subheader("Remove Stock")
        col_del, col_btn = st.columns([2, 1])
        to_del = col_del.selectbox("Select stock to remove:", st.session_state.portfolio['Symbol'])
        if col_btn.button("🗑️ Delete Selected", use_container_width=True):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != to_del]
            save_data(st.session_state.portfolio)
            st.rerun()

st.sidebar.divider()
st.sidebar.caption("v2.7 | Master Portfolio Management")