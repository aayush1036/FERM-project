import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from datetime import datetime as dt 
from dateutil.relativedelta import relativedelta
import numpy as np 
import os 
import pandas_datareader.data as pdr
import yfinance as yf 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from tabulate import tabulate
yf.pdr_override()

class Portfolio:
    def __init__(self, stocks, period=1,save=True) -> None:
        self.stock_list = stocks
        self.__end_date = dt.today() - relativedelta(days=1)
        self.__start_date = self.__end_date - relativedelta(years=period)
        self.stock_list = [stock if stock.endswith('.NS') else f'{stock}.NS' for stock in self.stock_list]
        
        data = pdr.get_data_yahoo(self.stock_list, start = self.__start_date, end=self.__end_date, max_retries=1)['Close']
        
        weightage = 1/len(self.stock_list)
        returns_list = []
        
        for stock in self.stock_list:
           returns_list.append(data[stock]/data[stock].iloc[0]) 
        
        self.returns_df = pd.concat(returns_list, axis=1)
        self.returns_df.columns =  self.stock_list
        
        weighted_returns = (self.returns_df.copy() * weightage).sum(axis=1)        
        self.returns_df['PortfolioReturns'] = weighted_returns
               
        save_path = 'Data'
        if save:
            os.makedirs(save_path, exist_ok=True)
            data.to_csv(os.path.join(save_path, f'stock_{len(os.listdir(save_path))+1}.csv'), index=False)
        
        self.summary_df = pd.DataFrame()
        stocknames = []
        returns = []
        volatility = []
        for stock in self.stock_list:
            stocknames.append(stock)
            returns.append(self.returns_df[stock].mean())
            volatility.append(self.returns_df[stock].std())
        self.summary_df['Symbol'] = stocknames
        self.summary_df['Returns'] = returns 
        self.summary_df['Volatility'] = volatility
    
    def __get_nifty_returns(self, mean=True):
        nifty = pdr.get_data_yahoo('^NSEI', start=self.__start_date, end=self.__end_date, progress=False)['Close']
        returns = nifty/nifty.iloc[0]
        if mean:
            return returns.mean() * len(self.returns_df)  
        else:
            return returns
    
    def __get_excess_returns(self, rf=0.06):
        mkt_returns = self.__get_nifty_returns(mean=False)
        portfolio_returns = self.returns_df['PortfolioReturns']
        beta = np.polyfit(x=mkt_returns, y=portfolio_returns, deg=1)[0]
        excess_returns = portfolio_returns.mean() - beta*(mkt_returns.mean() - rf)
        return beta, excess_returns
     
    def see_frontier(self):
        _, ax = plt.subplots(figsize=(16,6))
        ax.scatter(self.summary_df['Volatility'], self.summary_df['Returns'])
        min_risk_idx = self.summary_df['Volatility'].idxmin()
        stock_name = self.summary_df.loc[min_risk_idx]['Symbol']
        stock_return = self.summary_df.loc[min_risk_idx]['Returns']
        stock_volatility = self.summary_df.loc[min_risk_idx]['Volatility']
        ax.axhline(y=stock_return, ls='--', label=f'Return -> {stock_return:.3%}', color='blue')
        ax.axvline(x=stock_volatility, ls='--', label = f'Volatility -> {stock_volatility:.3%}', color='red')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Returns')
        ax.set_title(f'Efficient portfolio frontier (ideal portfolio = {stock_name})')
        plt.legend()
        plt.show()
    
    def describe(self, rf=0.06):
        portfolio_return = self.returns_df['PortfolioReturns'].mean() * len(self.returns_df)
        portfolio_volatility = self.returns_df['PortfolioReturns'].std() * np.sqrt(len(self.returns_df))
        market_returns = self.__get_nifty_returns()
        beta, excess_return = self.__get_excess_returns()
        sharpe = (portfolio_return - rf)/portfolio_volatility
        sharpe = sharpe * np.sqrt(len(self.returns_df))
        treynors = (portfolio_return-rf)/beta
        jensens = portfolio_return - (rf + beta * (market_returns-rf))
        information_ratio = (portfolio_return - rf)/excess_return
        
        info= {
            'Sharpe Ratio':[sharpe],
            'Treynors Ratio':[treynors],
            'Jensens Alpha':[jensens],
            'Information Ratio':[information_ratio]
        }
        
        print(tabulate(info, headers='keys'))
        return info
    
    def find_clusters(self):
        X = self.summary_df[['Returns','Volatility']].dropna().values 
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        inertia = []
        for k in range(1,10):
            model = KMeans(n_clusters=k)
            model.fit(X)
            inertia.append(model.inertia_)
        inertia = np.array(inertia)
        _, ax = plt.subplots(figsize=(16,6))
        ax.plot(list(range(1,10)),inertia)
        ax.set_title('Deciding optimal number of clusters')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Inertia')
        plt.show()
    
    def cluster(self,k):
        X = self.summary_df[['Returns','Volatility']].dropna().values
        self.cluster_df = self.summary_df.dropna().copy()
        pipeline = Pipeline(steps=[
            ('Scaler', MinMaxScaler()),
            ('Clustering', KMeans(n_clusters=k))
        ])
        pipeline.fit(X)
        preds = pipeline.predict(X)
        self.cluster_df['Cluster'] = preds 
        self.clustered = True 
        
    def visualize_clusters(self):
        if not self.clustered:
            raise NotImplementedError('Please cluster your data first')
        _, ax = plt.subplots(figsize=(16,6))
        sns.scatterplot(data=self.cluster_df, x='Volatility', y='Returns', hue='Cluster')
        ax.set_title('Cluster plot')
        ax.set_ylabel('Returns')
        ax.set_xlabel('Volatility')
        plt.show()
    