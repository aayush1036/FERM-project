import matplotlib.pyplot as plt 
import seaborn as sns 
import pandas as pd 
from datetime import datetime as dt 
from dateutil.relativedelta import relativedelta
import numpy as np 
import os 
import pandas_datareader.data as pdr
import yfinance as yf 
from scipy.special import softmax
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
yf.pdr_override()

class Portfolio:
    def __init__(self, stocks_path, period=1,save=True) -> None:
        self.stock_df = pd.read_csv(stocks_path)
        self.stock_df['Symbol'] = self.stock_df['Symbol'].apply(lambda x: x if x.endswith('.NS') else f'{x}.NS')
        stock_list = self.stock_df['Symbol'].tolist()
        end_date = dt.today() - relativedelta(days=1)
        start_date = end_date - relativedelta(years=period)
        
        data = pdr.get_data_yahoo(stock_list, start = start_date, end=end_date, max_retries=1)['Close']
        # Dropping stocks which have null values in their close columns
        data.dropna(axis=1, inplace=True)
        
        stock_list = data.columns
        returns_list = []
        
        for stock in stock_list:
           returns_list.append(data[stock].pct_change()) 
        
        self.returns_df = pd.concat(returns_list, axis=1)
        self.returns_df.columns =  stock_list
        
        save_path = 'Data/stocks'
        if save:
            os.makedirs(save_path, exist_ok=True)
            version = len(os.listdir(save_path)) + 1 
            data.to_csv(os.path.join(save_path, f'stock_{version}.csv'), index=False)
        
        self.summary_df = pd.DataFrame()
        stocknames = []
        returns = []
        volatility = []
        for stock in stock_list:
            stocknames.append(stock)
            returns.append(self.returns_df[stock].mean())
            volatility.append(self.returns_df[stock].std())
        self.summary_df['Symbol'] = stocknames
        self.summary_df['Returns'] = returns 
        self.summary_df['Volatility'] = volatility
        self.summary_df['Sharpe'] = self.summary_df['Returns'] / self.summary_df['Volatility']
        self.summary_df.sort_values(by='Sharpe', inplace=True, ascending=False)

        print(f'Portfolio(stocks_path={os.path.abspath(stocks_path)}, period={period}, save={save})')
    
    def monte_carlo(self, nsim=1000, topn=20, save_weights=True):
        stocks = self.summary_df['Symbol'].tolist()[:topn]
        df = self.returns_df[stocks]
        weights = np.random.random(size=(topn, nsim))
        weights = softmax(weights, axis=0)
        simulation = np.matmul(df, weights)
        weights_df = pd.DataFrame(weights).transpose()
        weights_df.columns = stocks
        weights_df['Simulation'] = range(1,nsim+1)
        
        if save_weights: 
            save_path = os.path.join('Data', 'weights')
            os.makedirs(save_path, exist_ok=True)
            version = len(os.listdir(save_path)) + 1 
            df_path = os.path.join(save_path, f'weights_{version}.csv')       
            weights_df.to_csv(df_path, index=False)
        
        simulation_df = pd.DataFrame({
            'Simulation':range(1,nsim+1),
            'Returns':simulation.mean(),
            'Volatility':simulation.std()
        })
        simulation_df['Sharpe'] = simulation_df['Returns']/simulation_df['Volatility']
        return simulation_df
     
    def see_frontier(self,ax=None):
        
        df = self.monte_carlo().copy()
        if ax is None:
            _, ax = plt.subplots(figsize=(16,6))
        ax.scatter(df['Volatility'], df['Returns'])
        max_sharpe_idx = df['Sharpe'].idxmax()
        stock_name = df.loc[max_sharpe_idx]['Simulation']
        stock_return = df.loc[max_sharpe_idx]['Returns']
        stock_volatility = df.loc[max_sharpe_idx]['Volatility']
        ax.axhline(y=stock_return, ls='--', label=f'Return -> {stock_return:.3%}', color='blue')
        ax.axvline(x=stock_volatility, ls='--', label = f'Volatility -> {stock_volatility:.3%}', color='red')
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Returns')
        ax.set_title(f'ideal portfolio = {stock_name}')
        ax.legend()    
    
    def find_clusters(self,ax=None):
        X = self.summary_df[['Returns','Volatility']].values 
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        inertia = []
        for k in range(1,10):
            model = KMeans(n_clusters=k)
            model.fit(X)
            inertia.append(model.inertia_)
        inertia = np.array(inertia)
        if ax is None:
            _, ax = plt.subplots(figsize=(16,6))
        ax.plot(list(range(1,10)),inertia)
        ax.set_title('Deciding optimal number of clusters')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Inertia')
        plt.show()
    
    def cluster(self,k):
        X = self.summary_df[['Returns','Volatility']].values
        self.cluster_df = self.summary_df.copy()
        pipeline = Pipeline(steps=[
            ('Scaler', MinMaxScaler()),
            ('Clustering', KMeans(n_clusters=k))
        ])
        pipeline.fit(X)
        preds = pipeline.predict(X)
        self.cluster_df['Cluster'] = preds 
        self.clustered = True 
        
    def visualize_clusters(self, ax=None):
        if not self.clustered:
            raise NotImplementedError('Please cluster your data first')
        if ax is None:
            _, ax = plt.subplots(figsize=(16,6))
        sns.scatterplot(data=self.cluster_df, x='Volatility', y='Returns', hue='Cluster')
        ax.set_title('Cluster plot')
        ax.set_ylabel('Returns')
        ax.set_xlabel('Volatility')
        plt.show()
    
    def infer_clusters(self, top=5):
        df = pd.merge(self.cluster_df, self.stock_df, on='Symbol')
        df.drop(['Company Name', 'Series', 'ISIN Code'], axis=1, inplace=True)
        cluster_dict = {cluster:
            pd.DataFrame(df[df['Cluster'] == cluster]['Industry'].value_counts()[:top]) 
            for cluster in df['Cluster'].unique()}
        return cluster_dict