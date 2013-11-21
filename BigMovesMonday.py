import numpy as np

#Adapted from 
#http://www.quantifiedstrategies.com/big-moves-on-mondays/
R_P = 1 #Refresh everyday
W_L = 25 #Calculate average from 25 days

def initialize(context):
    #SPY
    context.sids = [sid(8554)]
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.25,price_impact=0))
    set_commission(commission.PerShare(cost=0.05))
    
def handle_data(context,data):
    # get data
    priceData = get_data(data,context.sids)
    if priceData == None:
       return
    
    #Price data
    h = priceData['high']
    l = priceData['low']
    p = priceData['price']
    
    HLRange = np.zeros(W_L)
    
    #Get date information    
    date = data[data.keys()[0]].datetime
    
    #TODAY IS MONDAY
    if date.weekday() == 0:
        #Calculate 25 days HLRange 
        for i in range(0,len(p) - 2):
          HLRange[i] = abs(h[i]-l[i])
            
        #Average HLRange
        avgHLRange = np.average(HLRange) #Calculate the 25 days avg
        
        #Today's change
        td_change = (data[sid(8554)].price - p[len(p) - 2]) #Calculate SPY's today change
        
        #If price went down today
        if td_change < 0:
            #If the change is at least 50% HLRange
            if abs(td_change[0]) >= 0.5*avgHLRange:
                #Go all in
                Allin_amount = context.portfolio.cash/data[sid(8554)].price
                order(sid(8554), Allin_amount)
    
    #Today is thursday
    if date.weekday() == 4:
        #Exit all
        current_shares = context.portfolio.positions[sid(8554)].amount
        order(sid(8554),-current_shares)

# set globals R_P & W_L above
@batch_transform(refresh_period = R_P, window_length = W_L + 1)
def get_data(datapanel,sids):
    h = datapanel['high'].as_matrix(sids)
    l = datapanel['low'].as_matrix(sids)
    p = datapanel['close_price'].as_matrix(sids)
    return {'high':h,'low':l,'price':p}