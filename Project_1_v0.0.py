# THIS CODE WILL READ MULTIPLE FILES
# import glob

# path = r'E:\Phase 2\Project as homework' # use your path
# all_files = glob.glob(path + "/*.csv")

# li = []

# for filename in all_files:
#     df = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df)

# frame = pd.concat(li, axis=0, ignore_index=True)
import pandas as pd
import numpy as np
import os

#READING FILES
stock = [file for file in os.listdir('E:\Phase 2\Project as homework')]

for file in stock:
    print(file)

All_stock = pd.DataFrame()

#BELOW LINE OF CODE WILL READ EACH FILE, REPLACE NULL WITH MEAN VALUE FOR EACH COLUMN
for file in stock:
    DF = pd.read_csv('E:\Phase 2\Project as homework/'+file)
    for i in DF.columns[DF.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
        DF[i].fillna(DF[i].mean(),inplace=True)
    All_stock = pd.concat([All_stock, DF])

All_Month.head()

All_stock.info()
All_stock['Symbol'].value_counts()
All_stock['Series'].value_counts()

a = All_stock.tail(1000)

All_stock.isnull().sum()

All_stock_1 = All_stock.drop(['Date','Series','Company Name','Industry','ISIN Code'], axis = 1)

a = All_stock_1.tail(1000)

All_stock_1.isnull().sum()

All_stock_na = All_stock_1[All_stock_1.isna().any(axis=1)]

# All_stock_null = All_stock_1[All_stock_1.isnull().any(axis=1)]

All_stock_1.dropna(inplace=True)

print(len(All_stock))
print(len(All_stock_1))

#CREATING TARGET VARIABLE############################################################

All_stock_1['Target'] = All_stock_1['Close'].shift(1)
All_stock_1['Target'] = All_stock_1['Target'].fillna(0)
All_stock_1 = All_stock_1.drop(['Prev Close'],axis=1)
a = All_stock_1.head(1000)

# All_stock_1 = pd.DataFrame(All_stock_1)
###########################################################################################
cor = All_stock_1.corr()

columns = cor[cor["Target"]>0.7]["Target"]
columns_1 = cor[cor["Target"]<(-0.65)]["Target"]

C = pd.concat([columns,columns_1],axis=1)
columns_list = C.index
columns_list

All_stock_corr = pd.DataFrame(All_stock_1,columns = columns_list)

All_stock_1 = All_stock_1.drop(['Symbol'],axis=1)
#######################################################################################

x = All_stock_1.drop(['Target'],axis=1)#.values
y = All_stock_1.iloc[:,-1]#.values

#######################################################################################
from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#######################################################################################
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

x = sc_x.fit_transform(x)

#######################################################################################
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

import statsmodels.api as sm
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

print("coeffiient",reg.coef_)
print("Intercept",reg.intercept_)

from sklearn.metrics import mean_squared_error, r2_score

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print('R Square : %.2f' % r2_score(y_test, y_pred))

import statsmodels.api as sm
x_1 = sm.add_constant(x)

regressor_OLS = sm.OLS(endog=y, exog=x_1).fit()
regressor_OLS.summary()

x_opt = x_1[:, [0,1,2,5,6,9,10]]

regression_ols = sm.OLS(endog=y, exog=x_opt).fit()
regression_ols.summary()


# #Backward Elimination
# import statsmodels.api as sm
# def backwardElimination(x, SL):
#     numVars = len(x[0])
#     temp = np.zeros((470384,11)).astype(int)
#     for i in range(0, numVars):
#         regressor_OLS = sm.OLS(y, x).fit()
#         maxVar = max(regressor_OLS.pvalues).astype(float)
#         adjR_before = regressor_OLS.rsquared_adj.astype(float)
#         if maxVar > SL:
#             for j in range(0, numVars - i):
#                 if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                     temp[:,j] = x[:, j]
#                     x = np.delete(x, j, 1)
#                     tmp_regressor = sm.OLS(y, x).fit()
#                     adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                     if (adjR_before >= adjR_after):
#                         x_rollback = np.hstack((x, temp[:,[0,j]]))
#                         x_rollback = np.delete(x_rollback, j, 1)
#                         print (regressor_OLS.summary())
#                         return x_rollback
#                     else:
#                         continue
#     regressor_OLS.summary()
#     return a
 
# SL = 0.05
# X_opt = x#[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10]]
# X_Modeled = backwardElimination(X_opt, SL)
 
#DATA VISUALIZATION##################################################################
import matplotlib.pyplot as plt
import seaborn as sns

closing_prz = All_stock.groupby('Symbol').sum()

stk = [stk for stk, df in All_stock.groupby('Symbol')]
plt.bar(stk,closing_prz['Close'])
plt.xticks(stk, rotation = 'vertical', size = 8)
plt.ylabel('Closing price of stock')
plt.xlabel('Symbol of stock')
plt.show()

open_prz = All_stock.groupby('Symbol').sum()

stk = [stk for stk, df in All_stock.groupby('Symbol')]
plt.bar(stk,closing_prz['Open'])
plt.xticks(stk, rotation = 'vertical', size = 8)
plt.ylabel('Closing price of stock')
plt.xlabel('Symbol of stock')
plt.show()

#THIS CODE IS CONSUMING TOO MUCH MEMORY#################################################
# sns.countplot(x=All_stock_1['Open'],hue=All_stock_1['Symbol'], data = All_stock_1,palette='Set1')
# plt.title("Bar chart of Open prie of stock colored by stock symbol", fontsize=17)
# plt.xlabel("Purpose", fontsize=15)

All_stock['Symbol'].value_counts()

plt.hist(All_stock[All_stock['Symbol']=='TELCO']['Volume'],bins=30,alpha=0.5,color='red', label='TELCO Closing')
plt.hist(All_stock[All_stock['Symbol']=='MUNDRAPORT']['Volume'],bins=30,alpha=0.5,color='green', label='MUNDRAPORT Closing')
plt.hist(All_stock[All_stock['Symbol']=='TITAN']['Volume'],bins=30,alpha=0.5,color='yellow', label='TITAN Closing')
plt.hist(All_stock[All_stock['Symbol']=='ZEEL']['Volume'],bins=30,alpha=0.5,color='orange', label='ZEEL Closing')
plt.legend()
plt.xlabel('volume')
plt.title('STOCK VOLUME')
plt.show()


# All_Stock_Telco = All_stock_1[All_stock_1['Symbol']=='TELCO']
    
# # plt.figure(figsize=(15,8))
# sns.distplot(All_Stock_Telco['Open'],color = 'blue', label = 'Open prz')
# sns.distplot(All_Stock_Telco['Close'],color = 'r', label = 'Closing prx')
# sns.distplot(All_Stock_Telco['Volume'],color = 'g', label = 'Volm')
# sns.distplot(All_Stock_Telco['Trades'],color = 'orange', label = 'trades')
# plt.legend()
# plt.show()


plt.figure(figsize=(10,6))
sns.countplot(x=All_stock['Trades'],hue=All_stock['Symbol'], data = All_stock_1,palette='Set1')
plt.title("Bar chart of Trades of stocks", fontsize=17)
plt.xlabel("Trades", fontsize=15)

l=list(All_stock_1.columns)
l[0:len(l)-2]

# #THIS CODE WILL PLOT THE BOXPLOT/SCATTERPLOT FOR ALL INDEPENDENT VARIABLES AGAINST DEPENDENT VARIABLES
# for i in range(len(l)-1):
#     sns.boxplot(x='Target',y=l[i], data=All_stock_1)
#     plt.figure()
    
# for i in range(len(l)-1):
#     sns.scatterplot(x='Target',y=l[i], data=All_stock_1)
#     plt.figure()

# PANDAS PROFILING
from pandas_profiling import ProfileReport
profile = ProfileReport(All_stock, title='Pandas Profiling Report', explorative=True)
profile.to_widgets()
profile.to_file("E:\Phase 2\Project as homework\All_stock.html")
