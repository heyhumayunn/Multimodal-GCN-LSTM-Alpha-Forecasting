import os
import requests
import pandas as pd
import yfinance as yf
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

class DataLoader:
    def __init__(self, ticker="NVDA"):
        self.ticker = ticker
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.base_url = "https://api.polygon.io/v2/reference/news"

    def fetch_news(self, start_date, end_date, limit=1000):
        """Fetches news from Polygon.io with pagination."""
        all_articles = []
        params = {
            "ticker": self.ticker,
            "published_utc.gte": start_date,
            "published_utc.lte": end_date,
            "limit": limit,
            "sort": "published_utc",
            "order": "desc",
            "apiKey": self.api_key
        }
        
        current_url = self.base_url
        print(f"Fetching news for {self.ticker}...")
        
        while True:
            response = requests.get(current_url, params=params if current_url == self.base_url else None)
            data = response.json()
            
            if response.status_code == 200:
                results = data.get('results', [])
                all_articles.extend(results)
                if 'next_url' in data:
                    current_url = data['next_url'] + f"&apiKey={self.api_key}"
                else:
                    break
            else:
                break
                
        df = pd.DataFrame(all_articles)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['published_utc']).dt.date
            # Simple cleaning
            df = df[['Date', 'title', 'article_url', 'description', 'publisher']]
            df['Source'] = df['publisher'].apply(lambda x: x.get('name') if isinstance(x, dict) else 'Unknown')
            df.rename(columns={'title': 'Headline', 'description': 'Summary'}, inplace=True)
        return df

    def fetch_financials(self, start_date, end_date):
        """Fetches OHLC data from YFinance."""
        df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = [c[0] for c in df.columns]
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df