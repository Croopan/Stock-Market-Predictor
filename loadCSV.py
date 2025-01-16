import yfinance as yf
# Define the stock ticker and date range
ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2023-01-01"

# Download historical stock data
stock_data = yf.download(ticker, start=start_date, end=end_date)


stock_data.to_csv("historical_stock_data.csv")