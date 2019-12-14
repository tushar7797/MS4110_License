# -*- coding: utf-8 -*-
"""permit

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Vulpf2v0Qc1LfDsyl479L-7OFCVIY6DW
"""

#Load the Drive helper and mount
from google.colab import drive

#This will prompt for authorization.
drive.mount('/content/drive')

"""**In order to run the code set "path" to the directory just above the .csv file**"""

#importing neceassary libraries
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import pickle
# %matplotlib inline
path = 'drive/My Drive/data_analytics/'

"""# Label Encoding and Preprocessing of Data"""

#loading the dataset
data = pd.read_csv(path + 'License.csv')

#copying original dataset in data1
data1=data[:]

#splitting location into two separate columns (longitude and latitude)
combined= data1['Location'].str.strip('()').str.split(', ', expand=True).rename(columns={0:'Latitude', 1:'Longitude'})
data1['Latitude']= combined['Latitude']
data1['Longitude']= combined['Longitude']

#converting to float type
data1['Latitude'] = data1['Latitude'].apply(pd.to_numeric, errors='coerce')
data1['Longitude'] = data1['Longitude'].apply(pd.to_numeric, errors='coerce')

#normnalizing the columns containing longitude, latitude and estimated cost 
data1['Longitude']=(data1['Longitude']- data1['Longitude'].min())/( data1['Longitude'].max()- data1['Longitude'].min())
data1['Latitude']=(data1['Latitude']- data1['Latitude'].min())/( data1['Latitude'].max()- data1['Latitude'].min())
data1['Estimated Cost']= (data1['Estimated Cost']- data1['Estimated Cost'].min())/( data1['Estimated Cost'].max()- data1['Estimated Cost'].min())

# converting categorical strings to label encoded data
cat = data1['Neighborhoods - Analysis Boundaries'].tolist()
encoded_data, mapping_index = pd.Series(cat).factorize()
print(type(encoded_data))
data1['Encoding Neighborhood']= pd.DataFrame(encoded_data)

cat = data1['Existing Use'].tolist()
encoded_data, mapping_index = pd.Series(cat).factorize()
print(type(encoded_data))
data1['Encoded Existing Use']= pd.DataFrame(encoded_data)

cat = data1['Proposed Use'].tolist()
encoded_data, mapping_index = pd.Series(cat).factorize()
print(type(encoded_data))
data1['Encoded Proposed Use']= pd.DataFrame(encoded_data)

z = {'approved':1, 'issued':1, 'complete':1, 'expired':1, 'revoked': 0, 'disapproved':0, 'cancelled': 0}
x = data1['Current Status']
output = x.map(z)

data1['Output']= pd.DataFrame(output)

data1['Encoded Existing Use'] = data1['Encoded Existing Use'].replace(-1, np.nan)
data1['Encoded Proposed Use'] = data1['Encoded Proposed Use'].replace(-1, np.nan)
data1['Encoding Neighborhood'] = data1['Encoding Neighborhood'].replace(-1, np.nan)

# removing redundant columns and copying it to data2 Dataframe
data2 = data1.drop(['Block', 'Lot','Permit Type Definition','Permit Number','Record ID','Street Number', 'Street Number Suffix', 'Street Name', 'Street Suffix', 'Unit Suffix', 'Location', 'Neighborhoods - Analysis Boundaries', 'Current Status', 'Existing Use','Proposed Use'], axis=1)

# Take a look at data2 
data2.head()

# Segregating Month-Day-year
from datetime import datetime
 
year = lambda x: datetime.strptime(x, "%m/%d/%Y" ).year
data2['Permit_year'] = data2['Permit Creation Date'].map(year)


month = lambda x: datetime.strptime(x, "%m/%d/%Y" ).month
data2['Permit_month'] = data2['Permit Creation Date'].map(month)


day = lambda x: datetime.strptime(x, "%m/%d/%Y" ).day
data2['Permit_day'] = data2['Permit Creation Date'].map(day)


year_1 = lambda x: datetime.strptime(x, "%m/%d/%Y" ).year
data2['Filed_year'] = data2['Filed Date'].map(year_1)


month_1 = lambda x: datetime.strptime(x, "%m/%d/%Y" ).month
data2['Filed_month'] = data2['Filed Date'].map(month_1)


day_1 = lambda x: datetime.strptime(x, "%m/%d/%Y" ).day
data2['Filed_day'] = data2['Filed Date'].map(day_1)

data2.drop('Filed Date', axis =1, inplace = True)
data2.drop('Permit Creation Date', axis =1, inplace = True)

data2.head()

data2 = data2.drop(['Filed_year', 'Filed_month','Filed_day'], axis=1)

"""# XGBoost with label encoded data"""

#importing library
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

#splitting of data2 and fitting models
data2.head(6)
y = data2['Output']
X = data2 
X = X.drop('Output', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, random_state = 46, test_size = 0.2)
xgb_model = XGBClassifier(scale_pos_weight = 0.01, n_estimators = 400, max_depth = 5 )
xgb_model.fit(X_train, y_train)

#getting test score
xgb_model.score(X_test, y_test)

# printing classification report for test dataset
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,xgb_model.predict(X_test)))

# printing confusion matrix
print(confusion_matrix(y_test, xgb_model.predict(X_test)))

"""# RandomForest with label encoded data"""

#importing imputer and setting up mean and mode imputation models
from sklearn.preprocessing import Imputer
imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_mode = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

#copying data2 into data3 for imputation changes
data3 = data2[:]
data3.head()

#splitting into x and y
X= data3.drop('Output', axis=1)
y= data3['Output']

#splitting x into mean and mode imputable data
X_mean= X.drop(['Permit Type','Supervisor District','Zipcode','Encoding Neighborhood','Encoded Existing Use','Encoded Proposed Use', 'Permit_year', 'Permit_month','Permit_day' ], axis=1)
X_mode= X[['Permit Type','Supervisor District','Zipcode','Encoding Neighborhood','Encoded Existing Use','Encoded Proposed Use', 'Permit_year', 'Permit_month','Permit_day']]

#mean imputation
imputer_mean.fit(X_mean)
X_mean2= imputer_mean.transform(X_mean)

#converting to dataframe
X_mean2= pd.DataFrame(X_mean2 ,columns= X_mean.columns)

#checking null values
X_mean2.isnull().sum()

#mode imputation
imputer_mode.fit(X_mode)
X_mode2= imputer_mode.transform(X_mode)

# converting to dataframe
X_mode2= pd.DataFrame(X_mode2 ,columns= X_mode.columns)

#checking null values
X_mode2.isnull().sum()

#concatenating mean and mode imputed data
X1 = pd.concat([X_mean2, X_mode2], axis=1)
data4 = pd.concat([X1 , y], axis=1)

#importing random forest
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()

# splitted data
X_train, X_test, y_train, y_test = train_test_split(X1,y, stratify = y, random_state = 46, test_size = 0.2)
rf.fit(X_train, y_train)

#getting score over test data
rf.score(X_test,y_test)

# classification report
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,rf.predict(X_test)))

# confusion matrix
confusion_matrix(y_test, rf.predict(X_test))





"""# Random Forest with upsampling"""

#splitting mean and mode imputed dataset
X_train, X_test, y_train, y_test = train_test_split(X1, y, stratify = y, random_state = 46, test_size = 0.2)

#concatenating xtrain and ytrain to data_new
data_new= pd.concat([X_train , y_train], axis=1)

data_new.head()

#resample function
from sklearn.utils import resample
# Separate majority and minority classes
df_majority = data_new[data_new.Output==1]
df_minority = data_new[data_new.Output==0]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= 136456,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.Output.value_counts()

df_upsampled.head()

from sklearn.ensemble import RandomForestClassifier
y_train_new = df_upsampled['Output']
X_train_new = df_upsampled.drop(['Output'],axis = 1)
# X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2, stratify = y2, random_state = 46, test_size = 0.2)
rf_upsample = RandomForestClassifier()

#fitting random forest
rf_upsample.fit(X_train_new, y_train_new)

#getting feature importance
rf_upsample.feature_importances_

# bar graph of feature importance
import matplotlib.pyplot as plt
plt.bar(range(len(rf_upsample.feature_importances_)),rf_upsample.feature_importances_)

#prediction 
rf_upsample.score(X_test , rf_upsample.predict(X_test))

#training score
rf_upsample.score(X_train , rf_upsample.predict(X_train))

#classification report 
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,rf_upsample.predict(X_test)))

#confusion matrix
print(confusion_matrix(y_test,  rf_upsample.predict(X_test)))

"""# One-hot encoding of Data

This section one-hot encodes data and processes the filled date and the location. The data is then stored in a file called 'data_merged.csv' which merges all the columns along with the one-hot encoding.
"""

data3 = data.copy()

data3.drop(['Permit Number', 'Permit Type Definition',
       'Permit Creation Date', 'Block', 'Lot', 'Street Number',
       'Street Number Suffix', 'Street Name', 'Street Suffix', 'Unit',
       'Unit Suffix','Record ID'], axis = 1, inplace = True)

j=0

permit_type_onehot = pd.get_dummies(data3['Permit Type'],dummy_na= True)
(a,b) = np.shape(permit_type_onehot)
permit_type_onehot.columns = [i +j for i in range(b)]
j = j +b

existing_use_onehot  =  pd.get_dummies(data3['Existing Use'],dummy_na= True)
(a,b) = np.shape(existing_use_onehot)
existing_use_onehot.columns = [i+j for i in range(b)]
j = j+b


proposed_use_onehot  =  pd.get_dummies(data3['Proposed Use'],dummy_na= True)
(a,b) = np.shape(proposed_use_onehot)
proposed_use_onehot.columns = [i+j for i in range(b)]
j = j+b


supervisor_district_onehot  =  pd.get_dummies(data3['Supervisor District'],dummy_na= True)
(a,b) = np.shape(supervisor_district_onehot)
supervisor_district_onehot.columns = [i+j for i in range(b)]
j = j+b

neighborhood_onehot  =  pd.get_dummies(data3['Neighborhoods - Analysis Boundaries'],dummy_na= True)
(a,b) = np.shape(neighborhood_onehot)
neighborhood_onehot.columns = [i+j for i in range(b)]
j = j+b

zipcode_onehot  =  pd.get_dummies(data3['Zipcode'],dummy_na= True)
(a,b) = np.shape(zipcode_onehot)
zipcode_onehot.columns = [i+j for i in range(b)]
j = j+b

data_merged = data3.drop(['Permit Type', 'Existing Use',  'Proposed Use', 'Supervisor District',
                          'Neighborhoods - Analysis Boundaries', 'Zipcode'], axis = 1)

data_merged = data_merged.join(permit_type_onehot)
data_merged = data_merged.join(existing_use_onehot)
data_merged = data_merged.join(proposed_use_onehot)
data_merged = data_merged.join(supervisor_district_onehot)
data_merged = data_merged.join(neighborhood_onehot)
data_merged = data_merged.join(zipcode_onehot)



print(data_merged.head())

data4=data.copy()
combined= data4['Location'].str.strip('()').str.split(', ', expand=True).rename(columns={0:'Latitude', 1:'Longitude'})
data4['Latitude']= combined['Latitude']
data4['Longitude']= combined['Longitude']
data4['Latitude'] = data4['Latitude'].apply(pd.to_numeric, errors='coerce')
data4['Longitude'] = data4['Longitude'].apply(pd.to_numeric, errors='coerce')
data4['Longitude']=(data4['Longitude']- data4['Longitude'].min())/( data4['Longitude'].max()- data4['Longitude'].min())
data4['Latitude']=(data4['Latitude']- data4['Latitude'].min())/( data4['Latitude'].max()- data4['Latitude'].min())
data_merged['Latitude'] = data4['Latitude']
data_merged['Longitude'] = data4['Longitude']
print(data_merged.columns)

from datetime import datetime

year_1 = lambda x: datetime.strptime(x, "%m/%d/%Y" ).year
data_merged['Filed_year'] = data_merged['Filed Date'].map(year_1)


month_1 = lambda x: datetime.strptime(x, "%m/%d/%Y" ).month
data_merged['Filed_month'] = data_merged['Filed Date'].map(month_1)


day_1 = lambda x: datetime.strptime(x, "%m/%d/%Y" ).day
data_merged['Filed_day'] = data_merged['Filed Date'].map(day_1)

data_merged.drop(['Filed Date', 'Location'], axis =1, inplace = True)

data_merged.drop(['Current Status'], axis =1, inplace = True)
print(data_merged.columns)

data_merged.to_csv(path +'merged_data.csv')

data_merged.shape

"""#Random Forest with new dataset"""

#importing new data
data5 = pd.read_csv(path +'merged_data.csv')

data5.head()

#mean and mode impute
from sklearn.preprocessing import Imputer
imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer_mode = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

data5= data5.drop("Unnamed: 0", axis=1)

imputer_mean.fit(data5)
X_new= imputer_mean.transform(data5)

X_new= pd.DataFrame(X_new ,columns= data5.columns)

data6 = pd.concat([X_new , y], axis=1)

data6.head()

X_train, X_test, y_train, y_test = train_test_split(X_new, y, stratify = y, random_state = 46, test_size = 0.2)

data_new2= pd.concat([X_train , y_train], axis=1)

from sklearn.utils import resample
# Separate majority and minority classes
df_majority = data_new2[data_new.Output==1]
df_minority = data_new2[data_new.Output==0]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples= 136456,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.Output.value_counts()

from sklearn.ensemble import RandomForestClassifier
y_train_new = df_upsampled['Output']
X_train_new = df_upsampled.drop(['Output'],axis = 1)
# X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2, stratify = y2, random_state = 46, test_size = 0.2)
rf_new = RandomForestClassifier()



rf_new.fit(X_train_new, y_train_new)

rf_new.feature_importances_

import matplotlib.pyplot as plt
plt.bar(range(len(rf_new.feature_importances_)), rf_new.feature_importances_)

arr= np.where(rf_new.feature_importances_>0.05)
#X_train_new.columns[2]

imp_features = X_train_new.columns[arr]
imp_features



rf_new.score(X_train , y_train)

rf_new.score(X_test , y_test)

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,rf_new.predict(X_test)))

print(confusion_matrix(y_test, rf_new.predict(X_test)))



import pickle

pickle.dump(rf_new, open(path + "model.sav", 'wb'))



"""# Hyperparamter Tuning"""

#Generate a graph that looks at the variation of Precision, Recall & F1-Score with the n_estimators HyperParameter
from sklearn.metrics import f1_score, precision_score, recall_score

f1_score_train = []
precision_train = []
recall_train = []

f1_score_test = []
precision_test = []
recall_test = []
n_estimators = [5,10,50,100,150,200]
for hp in n_estimators:
  rf_model_tune = RandomForestClassifier(n_estimators = hp)
  rf_model_tune.fit(X_train_new, y_train_new)
  
  f1_score_train.append(f1_score(y_train_new,rf_model_tune.predict(X_train_new)))
  precision_train.append(precision_score(y_train_new,rf_model_tune.predict(X_train_new)))
  recall_train.append(recall_score(y_train_new,rf_model_tune.predict(X_train_new)))
  
  f1_score_test.append(f1_score(y_test,rf_model_tune.predict(X_test)))
  precision_test.append(precision_score(y_test,rf_model_tune.predict(X_test)))
  recall_test.append(recall_score(y_test,rf_model_tune.predict(X_test)))

#Visualize the evaluation metrics
plt.style.use('ggplot')
plt.plot(n_estimators,f1_score_train, color = 'red', label = "Training Set F1 Score")
plt.plot(n_estimators,f1_score_test, color = 'blue', label = "Test Set F1 Score")
plt.legend()