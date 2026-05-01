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
MOCK_FILE = "mock_portfolio.csv"
LOG_FILE = "advisor_log.csv"

# --- DATA STORAGE ENGINE ---
def load_data(file, columns):
    if os.path.exists(file): 
        try: return pd.read_csv(file)
        except: return pd.DataFrame(columns=columns)
    return pd.DataFrame(columns=columns)

def save_data(df, file):
    df.to_csv(file, index=False)

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

# --- ANALYTICS ENGINE ---
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

def get_ai_analysis(symbol, name, tech_data, newbie_mode=False):
    news_items = tech_data.get('news', [])
    titles = [item.get('title') or "" for item in news_items[:7]]
    combined_news = "\n".join(filter(None, titles)) if titles else "No current news found."
    
    tone = "beginner-friendly with 'Good Vibes' focus" if newbie_mode else "professional and technical"
    prompt = f"""
    Act as a senior investment advisor. Analyze {name} ({symbol}).
    Style: {tone}.
    Current Data: RSI {tech_data['rsi']:.2f}, Trend {tech_data['trend']}.
    Recent News: {combined_news}
    Format: **[Score/10]** | **[REC]** | [One-sentence key insight]
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a professional investment advisor."},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except: return "Analysis currently unavailable."

# --- UI SETUP ---
st.set_page_config(page_title="StockWise Advisor Pro", page_icon="📈", layout="wide")

# Initialize Session States
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_data(PORTFOLIO_FILE, ["Symbol", "Quantity", "Purchase_Price"])
if 'mock_portfolio' not in st.session_state:
    st.session_state.mock_portfolio = load_data(MOCK_FILE, ["Symbol", "Quantity", "Watch_Price", "Date_Added"])

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("StockWise AI Pro")
    app_mode = st.selectbox("Menu", ["Dashboard", "Mock Simulator", "Analyst Good Vibes", "Performance Log", "Management"])
    st.divider()
    st.header("💡 Quick Guide")
    st.markdown("""
    **RSI (Momentum)**
    - Over 70: Hot (Careful!)
    - Under 30: Discount
    **Trend**
    - Bullish: Long-term growth
    - Bearish: Short-term decline
    """)

# --- DASHBOARD (REAL) ---
if app_mode == "Dashboard":
    st.title("My Real Portfolio 📊")
    if st.session_state.portfolio.empty:
        st.info("Your portfolio is empty. Add stocks in Management.")
    else:
        with st.spinner('Updating Market Data...'):
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

        m1, m2 = st.columns(2)
        m1.metric("Total Portfolio Value", f"${df['Value'].sum():,.2f}")
        m2.metric("Total Profit/Loss", f"${df['Gain'].sum():,.2f}", f"{((df['Gain'].sum() / (df['Value'].sum() - df['Gain'].sum())) * 100):.2f}%" if df['Value'].sum() != 0 else "0%")
        
        st.dataframe(df[['Symbol', 'Name', 'Quantity', 'Price', 'RSI', 'Trend', 'Gain']].style.format(precision=2), use_container_width=True)
        
        if st.button("Run Portfolio AI Audit"):
            for i, sym in enumerate(df['Symbol']):
                if metrics[i]:
                    with st.expander(f"Audit: {df['Name'].iloc[i]} ({sym})"):
                        analysis = get_ai_analysis(sym, df['Name'].iloc[i], metrics[i])
                        st.write(analysis)
                        if st.button(f"Log this analysis", key=f"log_{sym}"):
                            log_recommendation(sym, df['Name'].iloc[i], analysis, df['Price'].iloc[i])
                            st.toast("Saved to Log!")

# --- MOCK SIMULATOR ---
elif app_mode == "Mock Simulator":
    st.title("Mock Investment Simulator 🧪")
    st.write("Test your instincts with zero risk.")
    
    with st.expander("➕ Add New Simulation"):
        with st.form("new_mock"):
            ms = st.text_input("Ticker Symbol").upper().strip()
            mq = st.number_input("Virtual Quantity", value=10.0)
            mw = st.number_input("Target Entry Price", value=0.0)
            if st.form_submit_button("Start Simulation"):
                new_row = pd.DataFrame([{"Symbol": ms, "Quantity": mq, "Watch_Price": mw, "Date_Added": datetime.now().strftime("%Y-%m-%d")}])
                st.session_state.mock_portfolio = pd.concat([st.session_state.mock_portfolio, new_row], ignore_index=True)
                save_data(st.session_state.mock_portfolio, MOCK_FILE)
                st.rerun()

    if not st.session_state.mock_portfolio.empty:
        for idx, row in st.session_state.mock_portfolio.iterrows():
            data = get_tech_indicators(row['Symbol'])
            if data:
                gain = (data['price'] - row['Watch_Price']) * row['Quantity']
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 2, 1])
                    c1.markdown(f"### {row['Symbol']}")
                    c1.caption(f"Entry: ${row['Watch_Price']:.2f} | Now: ${data['price']:.2f}")
                    
                    c2.markdown(f"**Mock Profit:** :{'green' if gain >= 0 else 'red'}[${gain:,.2f}]")
                    c2.caption(f"Added on {row['Date_Added']}")
                    
                    if c3.button("💎 Buy for Real", key=f"conv_{idx}"):
                        st.session_state[f"buying_{idx}"] = True
                
                if st.session_state.get(f"buying_{idx}"):
                    with st.form(f"final_{idx}"):
                        st.write(f"Moving {row['Symbol']} to Real Holdings")
                        fq = st.number_input("Final Quantity", value=row['Quantity'])
                        fp = st.number_input("Final Price Paid", value=data['price'])
                        col_a, col_b = st.columns(2)
                        if col_a.form_submit_button("Confirm"):
                            # Move to Real
                            new_real = pd.DataFrame([{"Symbol": row['Symbol'], "Quantity": fq, "Purchase_Price": fp}])
                            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_real], ignore_index=True)
                            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
                            # Remove from Mock
                            st.session_state.mock_portfolio = st.session_state.mock_portfolio.drop(idx)
                            save_data(st.session_state.mock_portfolio, MOCK_FILE)
                            del st.session_state[f"buying_{idx}"]
                            st.rerun()
                        if col_b.form_submit_button("Cancel"):
                            del st.session_state[f"buying_{idx}"]
                            st.rerun()
    else:
        st.info("No simulations active. Add one above!")

# --- ANALYST GOOD VIBES ---
elif app_mode == "Analyst Good Vibes":
    st.title("Top Analyst 'Good Vibes' Picks 🌟")
    picks = {
        "MU": "Micron (AI Infrastructure Play)", 
        "SG": "Sweetgreen (High-Growth Retail)", 
        "HRMY": "Harmony Bio (Strong Medical Pipeline)", 
        "ASML": "ASML (The Chip-Maker's Chip-Maker)"
    }
    for sym, desc in picks.items():
        with st.expander(f"{sym} - {desc}"):
            if st.button(f"Analyze {sym} Opportunity"):
                data = get_tech_indicators(sym)
                if data:
                    analysis = get_ai_analysis(sym, sym, data, newbie_mode=True)
                    st.success(analysis)
                    log_recommendation(sym, sym, analysis, data['price'])

# --- PERFORMANCE LOG ---
elif app_mode == "Performance Log":
    st.title("Historical Advisor Log 📈")
    log_df = load_data(LOG_FILE, ["Date", "Symbol", "Name", "Price_At_Rec", "Analysis"])
    if log_df.empty:
        st.info("No logged recommendations yet.")
    else:
        st.dataframe(log_df.sort_values(by="Date", ascending=False), use_container_width=True)

# --- MANAGEMENT ---
elif app_mode == "Management":
    st.title("Settings & Management ⚙️")
    with st.form("manual_add"):
        st.subheader("Add Stock Manually")
        c = st.columns(3)
        s = c[0].text_input("Ticker").upper()
        q = c[1].number_input("Quantity")
        p = c[2].number_input("Price")
        if st.form_submit_button("Save to Real Portfolio"):
            new_row = pd.DataFrame([{"Symbol": s, "Quantity": q, "Purchase_Price": p}])
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
            st.rerun()

    if not st.session_state.portfolio.empty:
        st.divider()
        st.subheader("Remove Stock")
        td = st.selectbox("Select ticker to delete:", st.session_state.portfolio['Symbol'])
        if st.button("🗑️ Delete Permanently"):
            st.session_state.portfolio = st.session_state.portfolio[st.session_state.portfolio['Symbol'] != td]
            save_data(st.session_state.portfolio, PORTFOLIO_FILE)
            st.rerun()

st.sidebar.divider()
st.sidebar.caption("v3.3 | Complete Advisor Pro")