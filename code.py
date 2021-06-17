import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import math
from scipy import stats

with open('nba_2013.csv','r') as csvfile:
    nba_data = pd.read_csv(csvfile)
    
#LINEAR REGRESSION MULTI-VARIATE (least error)
#removing strings and null columns
X = nba_data.drop(['pos','player','bref_team_id','x3p.','ft.','season','season_end','pts','fg.','x2p.','efg.'], axis = 1) #features

#adding a numeric equivalent of pos string
temp_dict = {'SF':1,'C':2,'PF':3,'SG':4,'PG':5,'G':6,'F':7}
X['pos_num'] = nba_data.pos.map(lambda x: temp_dict[x]) 
y = nba_data['pts']

ig = [] #information gain
for x in X.columns:
    coeff = stats.pearsonr(X[x],y)[0] #stores the pearson's coeff
    if abs(round(coeff,0)) == 1:
        ig.append(x)
#ig has the list of the columns with the most correlation on the pts
X = X[ig]
train_X, val_X, train_y, val_y = train_test_split(X,y)
reg = linear_model.LinearRegression()
reg.fit(train_X,train_y)
pred = reg.predict(val_X)
li_mea = mean_absolute_error(pred, val_y)
print('the mean absolute error for linear regression is %0d' %li_mea)
comp = pd.DataFrame({'predicted': pred, 'actual': val_y,'diff':abs(pred - val_y)})
