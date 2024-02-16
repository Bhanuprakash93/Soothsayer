import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

class Stock:
    def __init__(self, name, ticker, start_date, end_date):
        self.name = name
        self.prices = yf.download(ticker, start=start_date, end=end_date)
        self.monthly_returns = self.prices["Close"].resample("M").last().pct_change()
        self.daily_returns = self.prices["Close"].pct_change()

    def cur_price(self, cur_date):
        return self.prices.loc[cur_date, "Close"]

    def monthly_ret(self, cur_date):
        formatted_date = cur_date.strftime('%Y-%m-%d')
        try:
            return self.monthly_returns.loc[formatted_date]
        except KeyError:
            closest_dates = self.monthly_returns.index[self.monthly_returns.index <= formatted_date]
            closest_date = closest_dates.max() if not closest_dates.empty else pd.NaT
            return self.monthly_returns.loc[closest_date] if not pd.isna(closest_date) else np.nan

    def daily_ret(self, cur_date):
        return self.daily_returns.loc[cur_date]

    def last_30days_price(self, cur_date):
        return self.prices.loc[cur_date - pd.to_timedelta(30, unit="D"):cur_date, "Close"]

def calculate_cagr(V_final, V_begin, t):
    return (((V_final / V_begin) ** (1/t)) - 1) * 100

def calculate_volatility(daily_returns):
    return np.sqrt(252) * np.std(daily_returns) * 100

def calculate_sharpe_ratio(mean_daily_returns, std_daily_returns):
    return np.sqrt(252) * (mean_daily_returns / std_daily_returns)        

def benchmark_strategy(nifty_data):
    nifty_returns = nifty_data["Close"].pct_change()
    cagr = calculate_cagr(nifty_data["Close"].iloc[-1], nifty_data["Close"].iloc[0], len(nifty_data))
    volatility = nifty_returns.std() * np.sqrt(252)
    sharpe_ratio = calculate_sharpe_ratio(nifty_returns.mean(), nifty_returns.std())
    return cagr, volatility, sharpe_ratio

def active_strategy(stocks_data, start_date, end_date):
    if start_date >= end_date:
        st.error("End date must be later than the start date.")
        return []

    portfolio = []
    for date in pd.date_range(start_date, end_date, freq="M"):
        monthly_returns = {}
        for stock_name, data in stocks_data.items():
            monthly_ret = data.monthly_ret(date)
            if not pd.isna(monthly_ret) and monthly_ret > 0:
                monthly_returns[stock_name] = monthly_ret.item()
        portfolio.append({date: monthly_returns})
    return portfolio

def calculate_allocation_metrics(portfolio, initial_equity, benchmark_metrics, nifty_data):
    benchmark_allocation = [initial_equity * (1 + benchmark_metrics[2] * x) for x in range(len(nifty_data))]
    
    active_allocation = []
    for item in portfolio:
        date = list(item.keys())[0]
        returns = item[date]
        
        active_stocks = [stock_name for stock_name, _ in returns.items()]
        num_active_stocks = len(active_stocks)

        if num_active_stocks > 0:
            equity_per_stock = initial_equity / num_active_stocks
            active_allocation.extend([equity_per_stock * (1 + returns[stock_name]) for stock_name in active_stocks])

    return benchmark_allocation, active_allocation
    
def calculate_active_strategy_metrics(active_allocation, initial_value):
    active_returns = np.diff(active_allocation) / active_allocation[:-1]
    cumulative_active_returns = np.cumprod(1 + active_returns) - 1

    final_value = initial_value * (1 + cumulative_active_returns[-1])
    num_periods = len(cumulative_active_returns)
    
    active_cagr = calculate_cagr(final_value, initial_value, num_periods)
    active_volatility = np.std(active_returns) * np.sqrt(252)
    active_sharpe_ratio = calculate_sharpe_ratio(np.mean(active_returns), np.std(active_returns))

    return active_cagr, active_volatility, active_sharpe_ratio

def main():
    st.title("Nifty 50 Stock Selection Strategy Dashboard")

    start_date = st.date_input("Select Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.date_input("Select End Date", pd.to_datetime("2024-01-01"))
    initial_equity = st.number_input("Enter Initial Equity", min_value=0, value=1000000)

    nifty_ticker = "^NSEI"
    nifty_data = yf.download(nifty_ticker, start=start_date, end=end_date)

    nifty_stocks = {
        'COALINDIA.NS': 'Coal India', 'UPL.NS': 'UPL', 'ICICIBANK.NS': 'ICICI Bank', 'NTPC.NS': 'NTPC',
        'HEROMOTOCO.NS': 'Hero MotoCorp', 'AXISBANK.NS': 'Axis Bank', 'HDFCLIFE.NS': 'HDFC Life',
        'BAJAJFINSV.NS': 'Bajaj Finserv', 'ONGC.NS': 'ONGC', 'APOLLOHOSP.NS': 'Apollo Hospitals',
        'SBIN.NS': 'SBI', 'KOTAKBANK.NS': 'Kotak Mahindra Bank', 'SBILIFE.NS': 'SBI Life',
        'BRITANNIA.NS': 'Britannia', 'SUNPHARMA.NS': 'Sun Pharma', 'BAJAJ-AUTO.NS': 'Bajaj Auto',
        'MARUTI.NS': 'Maruti Suzuki', 'LT.NS': 'Larsen & Toubro', 'RELIANCE.NS': 'Reliance Industries',
        'BPCL.NS': 'BPCL', 'TATACONSUM.NS': 'Tata Consumer', 'CIPLA.NS': 'Cipla',
        'M&M.NS': 'Mahindra & Mahindra', 'BAJFINANCE.NS': 'Bajaj Finance', 'ITC.NS': 'ITC',
        'ADANIPORTS.NS': 'Adani Ports', 'NESTLEIND.NS': 'Nestle', 'HDFCBANK.NS': 'HDFC Bank',
        'DIVISLAB.NS': 'Divi\'s Laboratories', 'TATAMOTORS.NS': 'Tata Motors', 'INDUSINDBK.NS': 'IndusInd Bank',
        'EICHERMOT.NS': 'Eicher Motors', 'HINDUNILVR.NS': 'Hindustan Unilever',
        'ASIANPAINT.NS': 'Asian Paints', 'DRREDDY.NS': 'Dr. Reddy\'s Laboratories',
        'ULTRACEMCO.NS': 'UltraTech Cement', 'BHARTIARTL.NS': 'Bharti Airtel', 'TITAN.NS': 'Titan',
        'TCS.NS': 'TCS', 'LTIM.NS': 'L&T Infotech', 'TATASTEEL.NS': 'Tata Steel', 'POWERGRID.NS': 'Power Grid',
        'HCLTECH.NS': 'HCL Technologies', 'TECHM.NS': 'Tech Mahindra', 'INFY.NS': 'Infosys',
        'WIPRO.NS': 'Wipro', 'JSWSTEEL.NS': 'JSW Steel', 'ADANIENT.NS': 'Adani Enterprises',
        'GRASIM.NS': 'Grasim Industries', 'HINDALCO.NS': 'Hindalco'
    }

    nifty_stock_objects = {}
    for ticker, name in nifty_stocks.items():
        stock = Stock(name=name, ticker=ticker, start_date=start_date, end_date=end_date)
        nifty_stock_objects[name] = stock

    benchmark_metrics = benchmark_strategy(nifty_data)

    st.header("Benchmark and Active Strategy Metrics")
    col1, col2 = st.columns(2)

    # Display Benchmark Metrics
    with col1:
        st.subheader("Benchmark Metrics")
        st.write("CAGR:", f"{benchmark_metrics[0] * 100:.2f}%")
        st.write("Volatility:", f"{benchmark_metrics[1] * 100:.2f}%")
        st.write("Sharpe Ratio:", f"{benchmark_metrics[2]:.4f}")

    portfolio = active_strategy(nifty_stock_objects, start_date, end_date)

    # Calculate active strategy metrics
    benchmark_allocation, active_allocation = calculate_allocation_metrics(portfolio, initial_equity, benchmark_metrics, nifty_data)
    
    # Calculate active strategy metrics with initial value
    active_cagr, active_volatility, active_sharpe_ratio = calculate_active_strategy_metrics(active_allocation, initial_equity)

    # Display Active Strategy Metrics
    with col2:
        st.subheader("Active Strategy Metrics")
        st.write("CAGR:", f"{active_cagr * 100:.2f}%")
        st.write("Volatility:", f"{active_volatility * 100:.2f}%")
        st.write("Sharpe Ratio:", f"{active_sharpe_ratio:.4f}")


    st.header("Active Strategy Portfolio")
    for item in portfolio:
        date = list(item.keys())[0]
        returns = item[date]
        st.subheader(f"{date.strftime('%B %Y')} Portfolio:")

        # Create a DataFrame for the portfolio to display as a table
        portfolio_df = pd.DataFrame(list(returns.items()), columns=['Stock Name', 'Monthly Return'])
        st.table(portfolio_df)

if __name__ == "__main__":
    main()
