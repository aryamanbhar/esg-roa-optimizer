import numpy as np
import pandas as pd
import os
import copy
import datetime
import yfinance as yf

'''
CHECK DOWNLOAD_DATA FOLDER IT HAS TWO FILES (DAILY STOCK DATA FOR DOW 30 COMPANIES)
OPEN --> Price at which the stock began trading when the market opened that day ($187.5)
HIGH --> Highest price reached by the stock during the day ($188.44)
LOW --> Lowest price at which the stock traded during the day ($183.89)
CLOSE --> Price at which the stock finished during the day ($184.73)
VOLUME --> Total number of shares traded during the day (82,488,700 shares)

E.G. JAN 2 2024 APPL
'''

# Example: Symbols of constituents in Dow Jones Industrial Average Index
# Please update the latest the constituents of the index based on your trading period.
DOW_30_TICKER = [
    "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "KO",
    "JPM", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW",
]


class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from neofinrl_config.py)
        end_date : str
            end date of the data (modified from neofinrl_config.py)
        ticker_list : list
            a list of stock tickers (modified from neofinrl_config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def fetch_data(self, proxy=None) -> pd.DataFrame:
        """Fetches data from Yahoo API
        Parameters
        ----------

        Returns
        -------
        `pd.DataFrame`
            7 columns: A date, open, high, low, close, volume and tick symbol
            for the specified stock ticker
        """
        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            temp_df = yf.download(
                tic, start=self.start_date, end=self.end_date, proxy=proxy
            )
            temp_df["tic"] = tic
            data_df = pd.concat([data_df, temp_df], axis=0)
        # reset the index, we want to use numbers as index instead of dates
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            data_df = data_df.drop(labels="adjcp", axis=1)
        except NotImplementedError:
            print("the features are not supported currently")
        # create day of the week column (monday = 0)
        # data_df["day"] = data_df["date"].dt.dayofweek
        # convert date to standard string format, easy to filter
        data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        # drop missing data
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)

        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df


def stockDataDownload_run():
    start_year = 2023
    end_year = 2023
    market_name = 'DOW' # 'SP500', 'DOW', 'HSI', 
    data_dir = './download_data'
    
    for yr in range(start_year, end_year+1):
        start_date = '{}-01-01'.format(yr)
        end_date = '{}-01-01'.format(yr+1)

        dname = '{}_{}.csv'.format(market_name, yr)
        mkt_dict = {
            'DOW': DOW_30_TICKER, 
            # Please add the constituents of other indices here
        }

        stock_lst = mkt_dict[market_name]
        if (market_name == 'SSE'):
            for idx in range(len(stock_lst)):
                old_mkt_code = stock_lst[idx].split('.')[1]
                stock_code = stock_lst[idx].split('.')[0]
                if old_mkt_code == 'XSHG':
                    new_mkt_code = 'SS'
                elif old_mkt_code == 'XSHE':
                    new_mkt_code = 'SZ'
                else:
                    raise ValueError('Unexpected market code: {}'.format(old_mkt_code))
                new_code = '{}.{}'.format(stock_code, new_mkt_code)
                stock_lst[idx] = new_code

        # print(stock_lst)
        # Example of stock_lst: ["AAPL", "JPM", "MSFT"]
        df = YahooDownloader(start_date = start_date,
                            end_date = end_date,
                            ticker_list = stock_lst).fetch_data()

        os.makedirs(data_dir, exist_ok=True)
        dpath = os.path.join(data_dir, dname)
        df.to_csv(dpath, header=True, index=False)
        print("Done [market: {}, year: {}]..".format(market_name, yr))


def mktIndexDownloader():
    yahoo_code = '^DJI' # SP500('^GSPC'), HSI(^HSI), NIKKRI225(^N225), DOW(^DJI)
    
    start_date = '2013-01-01'

    nextday_time = datetime.datetime.now() + datetime.timedelta(days=1)
    end_date = '{}-{}-{}'.format(nextday_time.year, str(nextday_time.month).zfill(2), str(nextday_time.day).zfill(2))
    
    data_dir = './download_data'
    fdir = os.path.join(data_dir, 'market_index')
    os.makedirs(fdir, exist_ok=True)

    fetch_start_date = start_date
    if yahoo_code == '^GSPC':
        mkt_name = 'SP500'
    elif yahoo_code == '^HSI':
        mkt_name = 'HSI'
    elif yahoo_code == '^N225':
        mkt_name = 'NIKKEI225'
    elif yahoo_code == '^DJI':
        mkt_name = 'DOW'
    elif yahoo_code == '^FTSE':
        mkt_name = 'FTSE350'
    elif yahoo_code == '^VIX':
        mkt_name = 'VIX'
    elif yahoo_code == '000300.SS':
        mkt_name = 'CSI300'
    elif yahoo_code == 'DX-Y.NYB':
        mkt_name = 'CRYPTO'
        fetch_start_date = '2021-09-25'
    else:
        raise ValueError('Unexpected market code: {}'.format(yahoo_code))
    fpath = os.path.join(fdir, '{}_index.csv'.format(mkt_name))

    content = YahooDownloader(start_date = start_date,
                        end_date = end_date,
                        ticker_list = [yahoo_code]).fetch_data()
    content['tic'] = mkt_name
    content = content[['date', 'tic', 'open', 'high', 'low', 'close', 'volume']]

    content.to_csv(fpath, header=True, index=False)
    print("Done {} market index from {} to {}...".format(mkt_name, start_date, end_date))
    

def main():
    
    stockDataDownload_run() # Download stock data
    mktIndexDownloader() # Download market index data


if __name__ == '__main__':
    main()