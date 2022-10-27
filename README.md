# FERM Project

Topic - Machine learning for portfolio diversification and allocation

Group members 
1. Aayushmaan Jain (J022)
2. Pratyush Patro (J047)
3. Devanshu Ramaiya (J050)
4. Ishani Shah (J067)
5. Amit Prajapati (J075)


Part 1: Machine Learning for portfolio diversificaion 

This part uses K-Means clustering to diversify the portfolio into three different clusters 
1. Low Risk High Return 
2. High Risk High Return
3. High Risk Low Return

Part 2: Monte Carlo Simulation for portfolio allocation 

This part uses monte carlo simulation to allocate weights to the top 20 stocks (according to sharpe ratio)<br>
For the purpose of computational efficiency, the simulation is done via matrix multiplicaion<br>

Data - $(n_{days}, n_{stocks})$<br>
Weights - $(n_{stocks}, n_{simulations})$<br>
Simuation matrix - $(n_{days}, n_{simulations})$

The simulation matrix contains the daily data for every simulation column wise which can then be aggregated to calculate the returns and volatility

<a href="https://colab.research.google.com/github/aayush1036/FERM-project/blob/master/project.ipynb">Demo on Google Colab</a>