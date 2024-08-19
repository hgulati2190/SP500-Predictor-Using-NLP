# salma_functions.py
import re
import requests
from bs4 import BeautifulSoup
import yfinance as yf
from datetime import datetime, timedelta
import nltk


def clean_text(text):
    nltk.data.path.append('/salma/nltk_data')
    nltk.download('punkt', download_dir='/salma/nltk_data')
    nltk.download('stopwords', download_dir='/salma/nltk_data')

    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

def collect_headlines():
    headlines = []
    cnbc_url = "https://www.cnbc.com/us-economy"
    response = requests.get(cnbc_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    cnbc_headlines = soup.find_all('a', class_='Card-title')
    for i, headline in enumerate(cnbc_headlines):
        if i >= 10:
            break
        headlines.append({'source': 'CNBC', 'text': headline.get_text(strip=True)})
    return headlines

def fetch_spy_data(date):
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d')
    if date.weekday() >= 5:
        date -= timedelta(days=(date.weekday() - 4))
    spy = yf.Ticker("SPY")
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
    data = spy.history(start=start_date, end=end_date)
    if not data.empty:
        return data[['Open']].reset_index().to_dict(orient='records')
    else:
        return []



