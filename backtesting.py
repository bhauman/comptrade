import pickle
import pytz
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.stats.api import ols
import datetime as dt
from dateutil import relativedelta

from zipline.algorithm import TradingAlgorithm
from zipline.transforms import batch_transform
from zipline.finance.slippage import FixedSlippage
from zipline.utils.factory import load_from_yahoo

class CorrelatedPairStrat():
    def __init__(self, trader, sym1, sym2, scale = 0.8, scalp = 0.8, trade_amount = 80000):
        self.trader = trader
        self.sym1 = sym1
        self.sym2 = sym2
        self.scale = scale
        self.scalp = scalp
        self.trade_amount = trade_amount
        self.regression_window = relativedelta.relativedelta(months=4)
        self.position_age_limit = relativedelta.relativedelta(months=2)
        m,t,c = self.do_regression()
        self.model = m
        self.theta = t
        self.c = c
        self.init_corr = self.corr()
        self.distances = []
        self.corr_history = []
        self.theta_history = []
        self.position_stack = []
        self.stack_limit = 100
        self.enabled = True

    def do_regression(self):
        end_date = self.trader.close_history.index[-1]
        start_date = end_date - self.regression_window
        history = self.trader.close_history #[start_date:]
        model =    ols(y=history[self.sym2], x=history.ix[:, [self.sym1]])
        theta =    model.beta[self.sym1]
        c =        model.beta["intercept"]
        return model, theta, c

    def corr(self):
        corr_df = self.trader.close_history[[self.sym1, self.sym2]].corr()
        return corr_df.ix[0,1]

    def healthy(self):
        return not (len(self.theta_history) > 0 and np.abs(self.theta - self.theta_history[-1]) > 0.2)

    def retire(self):
        """ retiring says sell for fair vaule if possible and don't enter new trades """
        self.scalp = 0.0
        self.enabled = False

    def distance(self, sym1_close, sym2_close):
        return (self.theta * sym1_close + self.c) - sym2_close

    def update_drift_histories(self):
        self.corr_history.append(self.corr())
        mod, theta, c = self.do_regression()
        self.theta_history.append(theta)

    def handle_data(self, data):
        self.update_drift_histories()
        self.age_out_positions(data)
        dist = self.distance(data[self.sym1]['price'], data[self.sym2]['price'])
        self.handle_trade(data, dist)
        self.distances.append(dist)

    def order_shares(self, data, sym, shares):
        self.trader.order(sym, shares)
        return {"sym": sym, 
                "price": data[sym].price, 
                "shares": shares}

    def age_out_positions(self, data):
        new_stack = []
        for p in self.position_stack:
            if data[self.sym1].datetime > (p["date"] + self.position_age_limit):
                print "AGING OUT POSITION"
                self.revert_position(p)
            else:
                new_stack.append(p)
        self.position_stack = new_stack
    
    def enter_position(self, data, amount, dist):
        """ if dist is positive the amount should be positive,
            if dist is negative the amount should be negative """
        if not self.allowed_to_enter_positions():
            print "NOT ALLOWED TO ENTER POSITION"
            return
        
        price2 = data[self.sym2].price
        shares2 = int(amount / price2)

        order_part2 = self.order_shares(data, self.sym2, shares2)
        order_part  = self.order_shares(data, self.sym1, -int(self.theta * shares2))
        position = {"amount": amount, 
                    "dist":   dist,
                    "date":   data[self.sym2].datetime,
                    "positions": [order_part, order_part2]}
        print position
        self.position_stack.append(position)

    def revert_position(self, position):
        for pos in position["positions"]:
            print "reverting", pos
            self.trader.order(pos["sym"], -pos["shares"])
            
    def revert_positions(self, data, dist):
        if len(self.position_stack) == 0:
            return
        top_pos = self.position_stack[-1]
        # close positions
        if ((top_pos["amount"] > 0 and dist <= (top_pos["dist"] - self.scalp)) or 
            (top_pos["amount"] < 0 and dist >= (top_pos["dist"] + self.scalp))):
            self.revert_position(top_pos)
            self.position_stack.pop()
            self.revert_positions(data, dist)
            
    def allowed_to_enter_positions(self):
        return (len(self.position_stack) < self.stack_limit) and self.enabled

    def handle_stack_positions(self, data, dist):
        self.revert_positions(data, dist)
        if len(self.position_stack) == 0:
            return
        top_pos = self.position_stack[-1]

        if top_pos["amount"] > 0 and  dist >= (top_pos["dist"] + self.scale):
            self.enter_position(data, self.trade_amount, dist)
        elif top_pos["amount"] < 0 and  dist <= (top_pos["dist"] - self.scale):
            self.enter_position(data, -self.trade_amount, dist)

    def handle_trade(self, data, dist):
        # when there is no position
        print "dist", dist, self.sym1, data[self.sym1].price, self.sym2, data[self.sym2].price, "stack_depth", len(self.position_stack)
        #print "position_total", self.portfolio.positions
        self.handle_stack_positions(data, dist)

        if len(self.position_stack) == 0 and np.abs(dist) > self.scale:
            if dist > self.scale:
                self.enter_position(data, self.trade_amount, dist)
            elif dist < -self.scale:
                self.enter_position(data, -self.trade_amount, dist)

class CompoundTrades(TradingAlgorithm):
    def initialize(self, close_history):
        self.close_history = close_history
        self.set_slippage(FixedSlippage())
        self.strats = []
        self.tick_count = 0

        self.pairs = [["AXP","WFC"],
                 ["APA","XOM"],
                 ["COP","SU"],
                 ["FCX","X"],
                 ["BK","C"],
                 ["CVX","OXY"],
                 ["MMM","SLB"],
                 ["CAT","UTX"],
                 ["GS","MS"],
                 ["F","GM"],
                 ["CAT","UNP"],
                 ["LVS", "WYNN"],
                 ["UPS","UTX"],
                 ["APA","WMB"]]
        for x in self.pairs:
            self.strats.append(CorrelatedPairStrat(self, 
                                                   x[0], 
                                                   x[1], 
                                                   scale = 0.8, 
                                                   scalp = 0.8, 
                                                   trade_amount = 10000))

    def append_to_close_history(self, data):
        columns = data.keys()
        prices = map(lambda k: data[k].price, columns)
        timestamp = data[columns[0]].datetime
        df = pd.DataFrame(index = [timestamp], columns = columns, data = [prices])
        self.close_history = self.close_history.append(df)

    def evaluate_and_replace_strat(self, strat):
        if not strat.healthy():
            strat.retire()
            return CorrelatedPairStrat(self, 
                                       strat.sym1, 
                                       strat.sym2, 
                                       scale = 0.3, 
                                       scalp = 0.3, 
                                       trade_amount = 10000)
        else:
            return False

    def evaluate_strats(self):
        if self.tick_count % 30 == 0:
            print self.tick_count, "EVALUATE_STRATS"
            for pair in self.pairs:
                strats_for_sym = filter(lambda x: x.sym1 == pair[0] and x.sym2 == pair[1], self.strats)
                assert len(strats_for_sym) > 0
                enabled_strats = filter(lambda x: x.enabled, strats_for_sym)
                assert len(enabled_strats) == 1
                enabled_strat = enabled_strats[0]
                new_s = self.evaluate_and_replace_strat(enabled_strat)
                if new_s:
                    self.strats.append(new_s)
                    assert not enabled_strat.allowed_to_enter_positions()

    def handle_data(self, data):
        self.tick_count = self.tick_count + 1
        self.append_to_close_history(data)
        #self.evaluate_strats()
        for s in self.strats:
            s.handle_data(data)

if __name__ == '__main__':
    data = pickle.load(open("pricedata.pkl",'r'))
    panel = pd.Panel(data)
    # closes_df = load_from_yahoo(stocks=[sym1, sym2])#panel.minor_xs('Adj Close')
    closes_df = panel.minor_xs('Adj Close')
    closes_df.index = closes_df.index.tz_localize(pytz.utc)
    
    

    learn_start = dt.datetime(2012, 10, 1) # - relativedelta.relativedelta(months = 9)
    learn_end =   dt.datetime(2013, 1, 1)  # - relativedelta.relativedelta(months = 9)

    reg_closes = closes_df[learn_start:learn_end]

    
    #train_closes = closes_df[learn_end:learn_end + relativedelta.relativedelta(months = 1)]
    train_closes = closes_df[learn_start:learn_end]
    #ser = pd.Series(train_closes[sym2].values, index= train_closes[sym1].values)

    #ser.plot(linestyle='None', marker="." )

    #regline = train_closes[sym1].values * model.beta[sym1] + c    
    #ser2 = pd.Series(regline, index= train_closes[sym1].values)
    #ser2.plot()
    
    #plt.plot(regline)
    #plt.figure()

    #trade_alg = TradeRegressionOscillation(sym1, sym2, theta, c, buy_thresh = 0.8)

    #trade_alg = TradeRegressionScaleIn(sym1, sym2, theta, c, scalp = 0.8, scale = 0.8)

    trade_alg = CompoundTrades(reg_closes)


    #((closes_copy[sym1] * theta + c) - closes_copy[sym2]).plot()

    #(10 * closes_copy[sym1].pct_change(1).cumsum()).plot(color="r")
    #(10 * closes_copy[sym2].pct_change(1).cumsum()).plot(color="g")
    #closes_copy["dist"].describe()
    
    #plt.figure()
    #print closes_copy.tail(50)

    results = trade_alg.run(train_closes)

    closes_copy = train_closes.copy()
    #trade_alg.strats[0].distances.plot()
    #closes_copy["corr"] = trade_alg.strats[0].corr_history
    #closes_copy["theta"] = trade_alg.strats[0].theta_history
    
    #closes_copy["dist"] = trade_alg.distances
    #closes_copy["dist"].plot()
    #plt.figure()
    #closes_copy["corr"].plot()
    #plt.figure()
    #closes_copy["theta"].plot()
    #plt.figure()
    results.portfolio_value.plot()



    print results.tail(100)
    plt.show()
