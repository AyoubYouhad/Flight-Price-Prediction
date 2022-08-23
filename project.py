import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor ,  ExtraTreesRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import datetime as dt
import missingno as msn
from sklearn.model_selection import train_test_split
# performing EDA
data=pd.read_excel('Data_Train.xlsx')
#print(data.head().transpose())
print(data.shape)
print(data.dtypes)
msn.bar(data,figsize=(20,8),color="#34495e",fontsize=12,labels=True)
plt.show()
data.dropna(inplace=True)
# the problem said that newDelhi is the same as Dehli , so we need to combine the two sources
#print(data['Destination'].value_counts())
def transform(city):
    if city=='New Delhi':
        return 'Delhi'
    else:
        return city
data['Destination']= data['Destination'].apply(transform)  

data['Journey_day'] = pd.to_datetime(data['Date_of_Journey'],format='%d/%m/%Y').dt.day
data['Journey_month'] = pd.to_datetime(data['Date_of_Journey'],format='%d/%m/%Y').dt.month
data.drop('Date_of_Journey',inplace=True,axis=1)

data['Dep_Time_minute'] = pd.to_datetime(data['Dep_Time']).dt.minute
data['Dep_Time_hour'] = pd.to_datetime(data['Dep_Time']).dt.hour
data.drop('Dep_Time',inplace=True,axis=1)

data['Arrival_minute'] = pd.to_datetime(data['Arrival_Time']).dt.minute
data['Arrival_hour'] = pd.to_datetime(data['Arrival_Time']).dt.hour
data.drop('Arrival_Time',inplace=True,axis=1)


duration = list(data['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i] + ' 0m'
        else:
            duration[i] = '0h ' + duration[i]
duration_hour = []
duration_min = []
for i in duration:
    h,m = i.split()
    duration_hour.append(int(h[:-1]))
    duration_min.append(int(m[:-1]))
data['duration_hour']=duration_hour
data['duration_min']=duration_min

#data.drop(['Duration','Route','Additional_Info'],axis=1,inplace=True)
print(data.columns)
#visualasation 
sns.catplot(x='Airline',y='Price',data=data.sort_values('Price',ascending=False),kind='boxen',aspect=3,height=6)
plt.show()
sns.catplot(x='Source',y='Price',data=data.sort_values('Price',ascending=False),kind='boxen',aspect=3,height=6)
plt.show()
sns.catplot(x='Destination',y='Price',data=data.sort_values('Price',ascending=False),kind='boxen',aspect=3,height=6)
plt.show()
# label encoding the'Total_Stops'columns
list=list(data['Total_Stops'])

print(data['Total_Stops'].value_counts())
def incode(stop):
    if stop=='non-stop':
        return 0
    elif stop=='1 stop':
        return 1
    elif stop=='2 stops':
        return 2
    elif stop=='3 stops':
        return 3
    elif stop=='4 stops':
        return 4
data['Total_Stops']= data['Total_Stops'].apply(incode)  
   
#apply one-hot-encodding
airline = data[['Airline']]
airline = pd.get_dummies(airline,drop_first=True)
source=data[['Source']]
source=pd.get_dummies(source,drop_first=True)
destination=data[['Destination']]
destination=pd.get_dummies(destination,drop_first=True)
data_train = pd.concat([data,airline,source,destination],axis=1)
data_train.drop(['Airline','Source','Destination'],axis=1,inplace=True)
#check the correlation 
sns.heatmap(data.corr(),cmap='viridis',annot=True)
plt.show()
print(data.columns)
#split the dependent variables from the independent variables
X=data.drop('Price',axis=1)
y=data['Price']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
Extra=ExtraTreesRegressor()
Extra.fit(X_train,y_train)
print(Extra.score(X_test,y_test))
print(Extra.feature_importances_)  
sns.barplot(x=data.columns,y=Extra.feature_importances_) 
plt.show()

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

forest_random = RandomizedSearchCV(estimator = RandomForestRegressor(), param_distributions = random_grid,
                               scoring='neg_mean_squared_error', n_iter = 10, cv = 5, 
                               verbose=1, random_state=42, n_jobs = -1)
forest_random.fit(X_train,y_train)
print(forest_random.best_params_)
print(forest_random.score(X_test,y_test))
    
    
    
