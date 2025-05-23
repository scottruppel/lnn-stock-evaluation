import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

class StockDataLoader:
    def __init__(self, tickers: List[str], start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
    
    def download_data(self) -> pd.DataFrame:
        """Download stock data using yfinance"""
        print(f"Downloading data for {self.tickers}...")
        self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        return self.data
    
    def get_closing_prices(self) -> Dict[str, np.ndarray]:
        """Extract closing prices for each ticker"""
        if self.data is None:
            self.download_data()
        
        prices = {}
        for ticker in self.tickers:
            prices[ticker] = self.data['Close'][ticker].values.reshape(-1, 1)
        
        return prices
