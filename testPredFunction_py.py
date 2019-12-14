
# coding: utf-8

# In[8]:


#importing neceassary libraries
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


def prediciton(filename,model):
    data = pd.read_csv(filename)
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
    
    data5 = data_merged
    
    from sklearn.preprocessing import Imputer
    imputer_mean = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer_mode = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
    
    imputer_mean.fit(data5)
    X_new= imputer_mean.transform(data5)
    X_new= pd.DataFrame(X_new ,columns= data5.columns)
    
    X = X_new.values
    
    y_pred = model.predict(X)
    
    return y_pred 


# In[10]:


import pickle

model = pickle.load(open("model.sav", 'rb'))

y_pred = prediciton("1.csv", model)


# In[12]:


np.bincount(y_pred)

