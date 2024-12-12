import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

class ESGAnalyzer:
    def __init__(self):
        pass
    
    def load_data(self):
        #Load stock data
        stock_data = pd.read_csv('DOW_2024.csv')
        stock_data['date'] = pd.to_datetime(stock_data['date'])
        
        #Load Sustainalytics ratings
        sust_data = pd.read_csv('sustainalytics_esg_score.csv')

        #Load MSCI ratings
        msci_data = pd.read_csv('msci_esg_individual_score.csv')
        
        #Merge ESG ratings with stock data
        combined_data = stock_data.merge(msci_data, on='tic', how='left')
        combined_data = combined_data.merge(sust_data, on='tic', how='left')
        
        return combined_data
        
    def create_features(self, data):
        #date is index for resampling
        data.set_index('date', inplace=True)
        
        features = pd.DataFrame()
        
        for ticker in data['tic'].unique():
            #filter based on specific ticker first
            stock_data = data[data['tic'] == ticker]
            
            #resample to monthly frequency by choosing the last date
            monthly_data = stock_data.resample('ME').agg({
                'close': 'last', 
            })
        
            monthly_data['returns'] = monthly_data['close'].pct_change()
            monthly_data['volatility'] = monthly_data['close'].rolling(window=2).std()
            
            monthly_data['tic'] = ticker
            
            features = pd.concat([features, monthly_data], ignore_index=True)
            
        features.reset_index(inplace=True)
            
        features.dropna(inplace=True)
        
        return features
        
def main():
    analyzer = ESGAnalyzer()
    data = analyzer.load_data()
    print(data.head())
    
    features = analyzer.create_features(data)
    print(features.head(15))
    
if __name__ == "__main__":
    main()
    