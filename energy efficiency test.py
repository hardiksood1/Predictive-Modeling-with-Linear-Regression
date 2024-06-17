#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# In[18]:


data = pd.read_csv(r"C:\Users\hp\Desktop\codealpha internship\predictive model with linear regression\ENB2012_data.csv")


# In[19]:


data.head()


# In[20]:


data.describe()


# In[21]:


data.isnull()


# In[22]:


X = data.drop(columns=['Y1', 'Y2'])
y = data['Y1']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)


# In[24]:


model = LinearRegression()


# In[25]:


model.fit(X_train, y_train)


# In[26]:


y_pred = model.predict(X_test)


# In[27]:


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')


# In[28]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Line')
plt.title('Actual vs Predicted Heating Load')
plt.xlabel('Actual Heating Load')
plt.ylabel('Predicted Heating Load')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:




