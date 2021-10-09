#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score


# In[17]:


df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(0)

df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.churn = (df.churn == 'yes').astype(int)


# In[23]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# In[24]:


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice',
               'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies',
               'contract', 'paperlessbilling', 'paymentmethod']
numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[25]:


def train(df, y, C=1.0):
    cat = df[categorical + numerical].to_dict(orient='rows')
    
    dv = DictVectorizer(sparse=False)
    dv.fit(cat)

    X = dv.transform(cat)

    model = LogisticRegression(solver='liblinear', C=C)
    model.fit(X, y)

    return dv, model


def predict(df, dv, model):
    cat = df[categorical + numerical].to_dict(orient='rows')
    
    X = dv.transform(cat)

    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[26]:


C = 1.0
n_splits = 5


# In[27]:


kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1)

scores = []
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train.churn.values
    y_val = df_val.churn.values
    
    dv, model = train(df_train, y_train, C = C)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    
print('C = %s %.3f +- %.3f' %(C, np.mean(scores), np.std(scores)))


# In[28]:


scores


# In[31]:


dv, model = train(df_full_train, df_full_train.churn.values, C = 1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# Save the model

# In[33]:


import pickle


# In[36]:


output_file = f'model_C={C}.bin'
output_file


# In[37]:


f_out = open(output_file, 'wb') #wb = write binary
pickle.dump((dv, model), f_out)
f_out.close() #dont forget to close the file


# In[38]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)
    #do something
#do others


# Load the model

# In[1]:


import pickle


# In[4]:


model_file = f'model_C=1.0.bin'


# In[5]:


with open(model_file, 'rb') as f_in: #to open a pickle file #rb = read bin
    dv, model = pickle.load(f_in)


# In[6]:


dv, model


# In[7]:


customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# In[9]:


X = dv.transform([customer])


# In[13]:


model.predict_proba(X)[0,1] #probability of the customer is going to churn


# In[ ]:




