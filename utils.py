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
yf.pdr_override()

class Portfolio:
    def __init__(self, stocks, period=1,save=True) -> None:
        self.stock_list = stocks
        self.period = period 
        self.end_date = dt.today()
        self.start_date = self.end_date + relativedelta(years=-self.period)
        print('Getting data')
        self.stock_list = [stock if stock.endswith('.NS') else f'{stock}.NS' for stock in self.stock_list]
        data = pdr.get_data_yahoo(self.stock_list, start = self.start_date, end=self.end_date, max_retries=1).Close.pct_change()
        returns = data.mean(axis=0)
        volatility = data.std(axis=0)
        self.stock_df = pd.DataFrame({
            'Symbol': self.stock_list,
            'Returns':returns,
            'Volatility':volatility
        })
        self.stock_df.reset_index(drop=True, inplace=True)
        self.save_path = 'Data'
        self.port_returns = data.mean(axis=1).dropna()
        self.nifty_returns = pdr.get_data_yahoo('^NSEI', start=self.start_date, end=self.end_date)['Close'].pct_change().dropna()
        self.beta = np.corrcoef(self.port_returns, self.nifty_returns)[0,1]/(self.port_returns.std()*self.nifty_returns.std())
        if save:
            os.makedirs(self.save_path, exist_ok=True)
            self.stock_df.to_csv(os.path.join(self.save_path, f'stock_{len(os.listdir(self.save_path))+1}.csv'), index=False)
    
    def see_frontier(self):
        _, ax = plt.subplots(figsize=(16,6))
        ax.scatter(self.stock_df['Volatility'], self.stock_df['Returns'])
        min_risk_idx = self.stock_df['Volatility'].idxmin()
        stock_name = self.stock_df.iloc[min_risk_idx, 0]
        stock_return = self.stock_df.iloc[min_risk_idx, 1]
        stock_volatility = self.stock_df.iloc[min_risk_idx, 2]
        ax.axhline(y=stock_return, ls='--', label=f'Return -> {stock_return:.3%}', color='blue')
        ax.axvline(x=stock_volatility, ls='--', label = f'Volatility -> {stock_volatility:.3%}', color='red')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Returns')
        ax.set_title(f'Efficient portfolio frontier (ideal portfolio = {stock_name})')
        plt.legend()
        plt.show()
    
    def describe(self, rf=0.06):
        portfolio_return = self.stock_df['Returns'].mean()
        portfolio_volatility = self.stock_df['Volatility'].mean()
        market_returns = self.nifty_returns.mean()
        excess_return = portfolio_return - self.beta*(market_returns-rf)
        sharpe = (portfolio_return - rf)/portfolio_volatility
        treynors = (portfolio_return-rf)/self.beta
        jensens = portfolio_return - (rf + self.beta * (market_returns-rf))
        information_ratio = (portfolio_return - rf)/excess_return
        return {
            'Sharpe Ratio':sharpe,
            'Treynors Ratio':treynors,
            'Jensens Alpha':jensens,
            'Information Ratio':information_ratio
        }
        
    def find_clusters(self):
        X = self.stock_df[['Returns','Volatility']].values 
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
        X = self.stock_df[['Returns','Volatility']].values
        pipeline = Pipeline(steps=[
            ('Scaler', MinMaxScaler()),
            ('Clustering', KMeans(n_clusters=k))
        ])
        pipeline.fit(X)
        preds = pipeline.predict(X)
        self.stock_df['Cluster'] = preds 
        self.clustered = True 
        
    def visualize_clusters(self):
        if not self.clustered:
            raise NotImplementedError('Please cluster your data first')
        _, ax = plt.subplots(figsize=(16,6))
        sns.scatterplot(data=self.stock_df, x='Volatility', y='Returns', hue='Cluster')
        ax.set_title('Cluster plot')
        ax.set_ylabel('Returns')
        ax.set_xlabel('Volatility')
        plt.show()
    