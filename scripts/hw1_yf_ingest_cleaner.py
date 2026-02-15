import pandas as pd
from pathlib import Path
from datetime import datetime
import re

def clean_current_prices():
    """
    Find and concatenate all current price csvs, and add the fed funds rate for each day
    """
    folder = Path("./data/raw")

    current_price_files = sorted(folder.glob("*_current_price_*.csv"))

    dataframes = []

    for file_path in current_price_files:
        df = pd.read_csv(file_path).T
        df.columns = df.iloc[0].values
        df = df.iloc[1:]

        ticker = file_path.stem.split("_")[0]
        df["ticker"] = ticker
        
        dataframes.append(df)

    current_prices = pd.concat(dataframes)
    current_prices.index = pd.to_datetime(current_prices.index).normalize().date

    # temporarily hardcode fed funds rate in the df as both days are same
    current_prices["fed_funds"] = 0.0364

    current_prices.to_csv(f"./data/cleaned/current_prices.csv")


def clean_options():
    """
    Find and concatenate all options data into DATA1 and DATA2 structure
    """
    folder = Path("./data/raw")

    option_files = sorted(folder.glob("*_options_*.csv"))

    dataframes = []

    for file_path in option_files:
        df = pd.read_csv(file_path)
        df["data_date"] = pd.to_datetime(file_path.stem.split("_")[-2])
        print(df.head())
        dataframes.append(df)

    options = pd.concat(dataframes)
    # I will use the mid-price (bid + ask) / 2 as the option's value, discarding all other info
    options = options[[
        "contractSymbol",
        "strike",
        "bid",
        "ask",
        "optionType",
        "expiration",
        "underlying",
        "data_date"
    ]].sort_values(by=["data_date", "expiration", "underlying", "optionType", "strike"])

    # to clean the data further, I will remove any options without posted quotes
    # this is primarily deep ITM options with less interest from traders
    print(f"Removing {len(options[(options["bid"] == 0) & (options["ask"] == 0)])} options with no posted quotes")

    options = options[(options["bid"] != 0) | (options["ask"] != 0)]

    options.to_csv("data/cleaned/options.csv")


if __name__ == "__main__":
    # clean_current_prices()
    clean_options()
