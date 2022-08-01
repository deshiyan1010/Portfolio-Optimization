import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class MPT:
    def __init__(self,tickers=None,fit_start="2019-01-01",fit_end="2019-12-31"):
        if tickers==None:
            self.tickers = ['AAPL', 'GOOGL', 'MSFT','AMZN' ,#Tech
                            'CCI','DLR','DRE', #Real estate
                            'ABT','ABBV', #Health
                            'AFL','JPM','GS', #Finance
                            'XOM','CVX','NEE' #Energy
                            ]
        else:
            self.tickers = tickers

        self.tickers.sort()
        data = yf.download(self.tickers,start=fit_start,end=fit_end)['Close']
        self.fit_start = fit_start
        self.fit_end = fit_end
        
        data = data.reset_index()
        data = data.drop(['Date'],axis=1)

        for c in data.columns:
            data[c] = data[c].astype('float32')


        # data = pd.DataFrame(columns=['A','B','C'])
        # data['A'] = [100+i for i in range(1,100)]
        # data['B'] = [100-i for i in range(1,100)]
        # data['C'] = [i*0.9+1 for i in range(1,100)]

        self.tickers = list(data.columns)

        self.data = data

        returns = data/data.shift(1)        
        self.meanRet = (returns.mean()**len(data)) -1
        self.covMat= returns.cov()*len(data)
        print(data.head(20))
        print(data.tail(20))
        print(self.covMat)
        print(self.meanRet)



    def get_weights(self,rMin,longs):
        N = len(self.covMat)
        pBarT = np.array(self.meanRet).reshape(-1,1)
        o = np.ones(N).reshape(-1,1)

        r = np.concatenate([pBarT,o],axis=1)
        F = np.concatenate([self.covMat,r],axis=1)
        
        zeros = np.zeros((2,2))
        b = np.concatenate([r.T,zeros],axis=1)
        F = np.concatenate([F,b],axis=0)

        x = np.zeros(N+2).reshape(-1,1)
        x[-2][0] = rMin
        x[-1][0] = 1

        #W = F^(-1)*x
        W = np.matmul(np.linalg.inv(F),x)[:-2]

        if longs==True:
            W = self.longify(W)

        return W

    def MVP(self,longs):
        N = len(self.covMat)
        pBarT = np.array(self.meanRet).reshape(-1,1)
        o = np.ones(N).reshape(-1,1)

        F = np.concatenate([self.covMat,o],axis=1)
        
        zeros = np.zeros((1,1))
        b = np.concatenate([o.T,zeros],axis=1)
        F = np.concatenate([F,b],axis=0)

        x = np.zeros(N+1).reshape(-1,1)
        x[-1][0] = 1

        #W = F^(-1)*x
        W = np.matmul(np.linalg.inv(F),x)[:-1]

        if longs==True:
            W = self.longify(W)

        return W


    def get_weights(self,rMin,longs):
        N = len(self.covMat)
        o = np.ones(N)
        covMatInv = np.linalg.inv(self.covMat)
        a = np.dot(self.meanRet.T,np.dot(covMatInv,self.meanRet))
        b = np.dot(self.meanRet.T,np.dot(covMatInv,o))
        c = np.dot(o.T,np.dot(covMatInv,o))

        W = 1/(a*c-b**2) *np.dot(covMatInv,(c*rMin-b)*self.meanRet+(a-b*rMin)*o)
        W = W.reshape(-1,1)

        if longs:
            W = self.longify(W)

        return W


    def mark(self,longs):
        low = 0
        high = 1
        e = 0.000001
        while high-low>e:
            mid = (low+high)/2
            
            l_er,l_ev = self.measure(self.get_weights(mid,longs))
            h_er,h_ev = self.measure(self.get_weights(mid+0.0001,longs))
            if l_ev>h_ev:
                low = mid
            else:
                high = mid
        
        er,ev = self.measure(self.get_weights(mid,longs))
        return er,ev,mid

    def for_max_vol(self,maxVol,longs):
        _,low,_ = self.mark(longs)

        if maxVol<low:
            raise Exception("Minimum volatility (mv): {}\n Expected volatility (ev): {}\nev<mv".format(low,maxVol))


        high = 4
        e = 0.000001
        while True:
            mid = (low+high)/2
            er,ev = self.measure(self.get_weights(mid,longs))
            if abs(ev-maxVol)<e:
                break
            if ev<maxVol:
                low = mid
            else:
                high = mid
        
        return self.get_weights(mid,longs)

    def for_fixed_return(self,ret,longs):
        return self.get_weights(ret,longs)


    def for_min_ret(self,minRet,longs):
        low,_,mvppoint = self.mark(longs)

        W = self.get_weights(10000,longs)
        er,ev = self.measure(W)
        if minRet>er:
            return W

        if minRet<low:
            return self.get_weights(low,longs)

        high = 1000000000
        e = 0.000001
        while True:
            mid = (low+high)/2
            # print(mid)
            er,ev = self.measure(self.get_weights(mid,longs))
            if abs(er-minRet)<e:
                break
            if er<minRet:
                low = mid
            else:
                high = mid
        
        return self.get_weights(mid,longs)



    def longify(self,W):
        W = np.where(W>=0,W,0)
        W = W/sum(W)
        return W

    def measurex(self,W):
        expected_return = np.matmul(self.meanRet,W)[0] 
        expected_volatility = np.matmul(np.matmul(W.T,self.covMat),W).iloc[0][0]
        return expected_return,(expected_volatility**0.5)


    def measure(self,W):
        expected_return = 0
        expected_volatility = 0

        for i in range(len(self.tickers)):
            expected_return += W[i][0]*self.meanRet[self.tickers[i]]
            for j in range(len(self.tickers)):
                expected_volatility += self.covMat[self.tickers[i]][self.tickers[j]]*W[i][0]*W[j][0]
        
        return expected_return,(expected_volatility**0.5)


    def plot(self,precision,samples,long):
        vol = []
        returns = []
        index = []
        for i in range(samples):
            r,v = self.measure(self.get_weights(i/precision,long))
            vol.append(v)
            returns.append(r)
            index.append(i/precision)

        fig, ax_left = plt.subplots()
        ax_right = ax_left.twinx()
        ax_left.plot(vol, linestyle = 'dotted',color='red')
        ax_right.plot(returns, color='green')
        plt.show()
    

    def plot_plotly(self,precision,samples,long,fileName='plot.html'):
        vol = []
        returns = []
        index = []
        weights_arr = []
        for i in range(samples):
            weights = self.get_weights(i/precision,long)
            r,v = self.measure(weights)
            vol.append(v)
            returns.append(r)
            index.append(i/precision)
            weights_arr.append("-".join(["( {} {:4f} )".format(t,w[0]*100) for t,w in zip(self.tickers,weights)]))

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(
            go.Scatter(x=index, y=vol, name="Volatility",hovertext=weights_arr),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(x=index, y=returns, name="Mean Return",hovertext=weights_arr),
            secondary_y=True,
        )

        fig.update_layout(
            title_text="Markowitz"
        )

        fig.update_xaxes(title_text="Samples")

        fig.update_yaxes(title_text="<b>Expected Volatility</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Expected Return</b>", secondary_y=True)

        fig.write_html(fileName)


    def plot_efficient_frontier(self,precision,samples,long,fileName='plot.html'):
        vol = []
        returns = []
        index = []
        weights_arr = []
        for i in range(samples):
            weights = self.get_weights(i/precision,long)
            r,v = self.measure(weights)
            vol.append(v)
            returns.append(r)
            index.append(i/precision)
            weights_arr.append("-".join(["( {} {:4f} )".format(t,w[0]*100) for t,w in zip(self.tickers,weights)]))

        fig = px.scatter(x=vol, y=returns,hover_name=weights_arr)

        fig.write_html(fileName)


    def plot_port_timeseries(self,n_days_in_future=365,rebalance_period_days=30):

        fs = self.fit_start
        fe = self.fit_end
        
        ts = (datetime.strptime(fe, '%Y-%m-%d')+ timedelta(days=1)).strftime('%Y-%m-%d')
        te = (datetime.strptime(fe, '%Y-%m-%d')+ timedelta(days=n_days_in_future)).strftime('%Y-%m-%d')


        price_action = []

        future = yf.download(self.tickers,start=ts,end=te)['Close']
        # print(future.head())
        # return
        initial_balance = 10000

        asset_held = {}
        for t in self.tickers:
            asset_held[t] = (initial_balance/len(self.tickers))/future.iloc[0][t]

        cW = np.zeros((len(self.tickers),1))
        # print(cW)
        c = 0
        for i in range(len(future)):
            print("Processing day {}...".format(i))
            if i%rebalance_period_days==0:
                print("Rebalancing...")
                mpt = MPT(self.tickers,fs,fe)
                W = mpt.for_min_ret(1,True)
                cW += W
                c+=1
                # print(mpt.measure(W),)
                fs = (datetime.strptime(fs, '%Y-%m-%d')+ timedelta(days=rebalance_period_days)).strftime('%Y-%m-%d')
                fe = (datetime.strptime(fe, '%Y-%m-%d')+ timedelta(days=rebalance_period_days)).strftime('%Y-%m-%d')

                curr_balance = 0
                for t in self.tickers:
                    curr_balance += asset_held[t]*future.iloc[i][t]

                for j,t in enumerate(self.tickers):
                    asset_held[t] = W[j][0]*curr_balance/future.iloc[i][t]



            current_worth = 0

            for tick in self.tickers:
                current_worth+=asset_held[tick]*future.iloc[i][tick]

            price_action.append(current_worth)

        df = pd.DataFrame()
        

        for t in self.tickers:
            df[t] = initial_balance/future.iloc[0][t] * np.array(future[t])

        df["MVP"] = price_action

        df['Date'] = future.index

        df.to_csv('data.csv')

        fig = px.line(df, x='Date', y=df.columns)
        
        fig['data'][-1]['line']['width'] = 7
        fig['data'][-1]['line']['color'] = 'red'
        fig.write_html('plotx.html')
        print(self.tickers)
        print((cW.reshape(-1))/c)
        # print(price_action)
        # plt.plot(price_action, linestyle = 'dashed')
        # plt.show()


if __name__=="__main__":
    mpt = MPT()

    mpt.plot_efficient_frontier(100,1000,True,fileName='long.html')
    mpt.plot_port_timeseries()


    
    # print(1)
    # print(mpt.measure(mpt.get_weights(0.1,True)))
    # print(mpt.measure(mpt.get_weights(0.1,False)))

    # print(mpt.measure(mpt.get_weights(10,True)))
    # print(mpt.measure(mpt.get_weights(10,False)))

    # print(2)
    # print(mpt.mark(True))
    # print(mpt.mark(False))
    
    # print(5)
    
    # mpt.plot_efficient_frontier(100,1000,False,fileName='long_short.html')


    # print(3)
    # print(mpt.measure(mpt.for_max_vol(0.11,True)))
    # print(mpt.measure(mpt.for_max_vol(0.11,False)))

    # print(mpt.measure(mpt.for_max_vol(0.13,True)))
    # print(mpt.measure(mpt.for_max_vol(0.13,False)))

    # print(4)
    # print(1,mpt.measure(mpt.for_min_ret(0.15,True)))
    # print(2,mpt.measure(mpt.for_min_ret(0.15,False)))

    # print(3,mpt.measure(mpt.for_min_ret(0.4,True)))
    # print(4,mpt.measure(mpt.for_min_ret(0.4,False)))



    # print(mpt.measure(mpt.min_return(0.917596,True)),"\n",mpt.min_return(10,True),"\n","\n")
    # print(mpt.measure(mpt.min_return(0.917596,False)),"\n",mpt.min_return(10,False))


    # print(mpt.measure(mpt.min_return(0.1,False)))
    # print(mpt.measure(mpt.min_return(0.7,True)))
    # print(mpt.measure(mpt.min_return(0.7,False)))
    # print(mpt.measure(mpt.for_fixed_vol(0.21,True)))
    # print(mpt.measure(mpt.for_fixed_vol(0.1895,True)))
    # mpt.plot_plotly(100,10000,True,'long_only.html')
    # mpt.plot_plotly(100,10000,False,'long_short.html')

    # print(mpt.plot_efficient_frontier(1000,5000,False,'ef_ls.html'))
    # W = np.array([[0.3],[0.4],[0.3]])

    # print(mpt.measure(W))
    # print(mpt.measurex(W))
    # print(mpt.covMat)
    # print(mpt.meanRet)



    # [[0.068795,  0.035584,  0.034704,  0.032451,  0.042755],
    # [0.035584,  0.052695,  0.033741,  0.032718,  0.031247],
    # [0.034704,  0.033741,  0.056394,  0.030040,  0.028353],
    # [0.032451,  0.032718,  0.030040,  0.039727,  0.028490],
    # [0.042755,  0.031247,  0.028353,  0.028490,  0.244164]]


    # print(mpt.mark(True))
    # print(mpt.measure(mpt.MVP(True)))
    # print(mpt.mark(False))
    # print(mpt.measure(mpt.MVP(False)))

    
