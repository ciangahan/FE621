import yfinance as yf
import pandas as pd

from datetime import datetime

def ingest_options(ticker:str, months:int = 3, option_day_offset:int = 0):
    """
    Ingest options data for a given ticker, including only traditional expiry dates (3rd friday of the month)
    @param ticker:str Ticker to ingest options chain for
    @param months:str Number of months to ingest
    @param option_day_offset:int Days to add to the option expiry to reach a 3rd friday of the month (ex VIX options)
    """
    tick = yf.Ticker(ticker)
    # filter out for options expiring on the third friday of the month (date between 15th and 21st)
    expirations = []
    for expiry in tick.options:
        expiry_dt = (pd.to_datetime(expiry) + pd.Timedelta(option_day_offset, 'day'))
        expiry_day = expiry_dt.day
        expiry_day_of_week = expiry_dt.day_of_week

        if (15 <= expiry_day <= 21) and expiry_day_of_week == 4:
            expirations.append(expiry)
    
    # then pick the first three of these for the next three months' data
    expirations = expirations[:months]

    for expiration in expirations:
        chain = tick.option_chain(expiration)
        calls = chain.calls
        puts = chain.puts
        calls['optionType'] = 'call'
        puts['optionType'] = 'put'
        
        options_df = pd.concat([calls, puts], ignore_index=True)
        # add expiration and ticker cols to avoid having to regex out of option id later
        options_df['expiration'] = expiration
        options_df['underlying'] = ticker
        
        # save to csv
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/raw/{ticker}_options_{expiration}_{timestamp}.csv"
        options_df.to_csv(filename, index=False)
        print(f"Saved {len(options_df)} rows to {filename}")
    
    # save current price to csv
    data = tick.history(period="1d", interval="1m")
    current_price = data.iloc[-1][["Close"]]

    filename = f"data/raw/{ticker}_current_price_{current_price.name}.csv"
    current_price.to_csv(filename)
    print(f"Saved {ticker} current price to {filename}")


if __name__ == "__main__":
    ingest_options("^VIX", option_day_offset=30)
    ingest_options("^SPX")
    ingest_options("TSLA")
