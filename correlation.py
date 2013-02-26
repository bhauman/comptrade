import datetime as dt
import math
import string
import pandas as pd
import numpy as np
from pandas.io.data import DataReader
from pandas.stats.api import ols
import matplotlib.pyplot as plt
import pickle

def cache_data():
    symbols = map(lambda x: x.strip(), open("cboe_syms.txt", 'r').readlines())
    
    data = {}
    for sym in symbols:
        print sym
        try:
            data[sym] = DataReader(sym, "yahoo")
        except:
            print "Error:  Not Found"
    pickle.dump(data, open("pricedata.pkl",'w'))

def get_data():
    data = pickle.load(open("pricedata.pkl",'r'))
    return pd.Panel(data)

def get_close_data(panel):
    return panel.minor_xs('Adj Close')

def correlations_above_thresh(correlations, thresh = 0.7):
    data = {}
    for cur_sym in correlations.index:
        column = correlations[cur_sym][correlations[cur_sym] > thresh][correlations[cur_sym] < 1]
        for sym in column.index:
            parts = [cur_sym, sym]
            parts.sort()
            key = string.join(parts, ":")
            data[key] = column[sym]

    series = pd.Series(data)
    series.sort()
    return series

def get_distances(sym, sym2, closes):
    model = ols(y=closes[sym2], x=closes.ix[:, [sym]])
    theta =    model.beta[sym]
    c =        model.beta["intercept"]
    return (theta * closes[sym] + c) - closes[sym2]

def get_oscillation_ratio(sym, sym2, close_data, start_date, end_date):
    closes = close_data.minor_xs('Adj Close')[start_date:end_date]
    pdistance = get_distances(sym, sym2, closes)
    diffs = np.abs(pdistance - pdistance.shift(1))
    return diffs.sum() / (pdistance.max() - pdistance.min())

def look_at_occilation(sym, sym2, close_data, start_date, end_date):
    closes = close_data.minor_xs('Adj Close')[start_date:end_date]
    pdistance = get_distances(sym, sym2, closes)
    
    diffs = np.abs(pdistance - pdistance.shift(1))

    pdistance.plot()
    print "diff to spread ratio", diffs.sum() / (pdistance.max() - pdistance.min())

    simulation = pd.DataFrame(index = closes.index)
    simulation[sym] = closes[sym]
    simulation[sym2] = closes[sym2]
    simulation["distance"] = pdistance
    simulation["diffs"] = diffs
    print simulation.head(50)
    print simulation["distance"].describe()


if __name__ == '__main__':
    data = pickle.load(open("pricedata.pkl",'r'))

    panel = pd.Panel(data)
    
    return_dataframe = panel.minor_xs('Adj Close').pct_change(1)
    
    # this method rocks the shit!
    # pd.scatter_matrix(return_dataframe)
#plt.figure()
    
    print "This is a correlation matrix of the equities"
#correlations = return_dataframe.corr()
# correlations.save("correlations.pkl")
    correlations = pickle.load(open("correlations.pkl"))
    
    high_correlations = correlations_above_thresh(correlations, 0.7)
    
    op_ratios = {}
    for syms in map(lambda x: x.split(":"), high_correlations.index):
        print syms
        op_ratios[syms[0] + ":" + syms[1]] = get_oscillation_ratio(syms[0],syms[1], panel, dt.datetime(2012, 10, 1), dt.datetime.now())

    results = pd.Series(op_ratios)
        
    results.sort()
        
    print results
    
    look_at_occilation("CAT", "UNP", panel, dt.datetime(2012, 10, 1), dt.datetime.now())
    plt.figure()
    look_at_occilation("F", "GM", panel, dt.datetime(2012, 10, 1), dt.datetime.now())
    plt.figure()
    look_at_occilation("GS", "MS", panel, dt.datetime(2012, 10, 1), dt.datetime.now())


#look_at_occilation("APA", "WMB", panel, dt.datetime(2012, 10, 1), dt.datetime.now())
#plt.figure()
#look_at_occilation("UPS", "UTX", panel, dt.datetime(2012, 10, 1), dt.datetime.now())
#plt.figure()
#look_at_occilation("LVS", "WYNN", panel, dt.datetime(2012, 10, 1), dt.datetime.now())

#plt.figure()
#look_at_occilation("CVX", "XOM", panel, dt.datetime(2012, 10, 1), dt.datetime.now())
#print correlations
    plt.show()

    exit()

#plt.figure()
# this is a 20day period rolling correlation of SPY to GOOG for the last 100 days
#roll_corr = pd.rolling_corr(return_dataframe, return_dataframe["SPY"], 20)['GOOG']
#small_slice = roll_corr[dt.datetime(2010,8,11):dt.datetime(2011,1,1)]
#small_slice.plot(kind="bar") # plot only the last 100 as a bar plot
#print "mean", small_slice.mean()
#plt.figure()

#roll_corr.hist() # create a more an informative histogram


    closes = panel.minor_xs('Adj Close')[dt.datetime(2012,2,1):]

    model = ols(y=closes['SPY'], x=closes.ix[:, ['GOOG']])


    closes['GOOG'].plot()
#closes['SPY'].plot(color="r")
    plt.figure()
    print model.beta["GOOG"]
    print model.beta["intercept"]
#model.beta.plot()

    spy_line = closes['GOOG'] * model.beta["GOOG"] + model.beta["intercept"]
    (spy_line - closes["SPY"]).plot()
    
    plt.show()




 
