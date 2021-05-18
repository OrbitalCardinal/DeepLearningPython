#!/usr/bin/env python
# coding: utf-8

# # Actividad 4.1 - Redes neuronales artificiales

# Construcción y entrenamiento de un modelo de red neuronal par la predicción de precios de casas utilizando el conjunto de datos de Boston_House_Prices.

# ## Analísis de datos

# ### Carga de datos

# In[4]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()


# In[5]:


data = pd.DataFrame(boston.data)


# In[6]:


data.head()


# Asigando precio y nombre de columnas

# In[7]:


data.columns = boston.feature_names


# In[17]:


data['PRICE'] = boston.target


# In[18]:


data.head()


# In[19]:


print(data.shape)


# Corroborando si existen valores nulos

# In[20]:


data.isnull().sum()


# Estadisticas descriptivas

# In[21]:


data.describe()


# In[22]:


data.info()


# Observando la distribución de los precios

# In[26]:


import seaborn as sns
sns.displot(data=data.PRICE)


# In[27]:


sns.boxplot(data=data.PRICE)


# Calculando los coefficientes de correlación

# In[28]:


correlation = data.corr()


# In[29]:


correlation.loc['PRICE']


# In[31]:


import matplotlib.pyplot as plt


# In[33]:


fig, axes = plt.subplots(figsize=(15,12))
sns.heatmap(correlation, square=True, annot=True)


# Observando las variables más llamativas obtenidas del analísis de correlación

# In[34]:


plt.figure(figsize=(20,5))
features = ['LSTAT', 'RM', 'PTRATIO']
for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = data[col]
    y = data.PRICE
    plt.scatter(x, y, marker='o')
    plt.title("Variation in House prices")
    plt.xlabel(col)
    plt.ylabel('"House prices in $1000"')


# ### Preparando los datos para el entrenamiento

# In[36]:


X = data.iloc[:,:-1]


# In[39]:


y = data.PRICE


# In[40]:


from sklearn.model_selection import train_test_split


# In[41]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 4)


# In[42]:


from sklearn.preprocessing import StandardScaler


# In[43]:


sc = StandardScaler()


# In[44]:


X_train = sc.fit_transform(X_train)


# In[45]:


X_test = sc.transform(X_test)


# ### Construcción del modelo de red neuronal

# In[50]:


import keras
from keras.layers import Dense, Activation, Dropout
from keras.models import Sequential


# In[51]:


model = Sequential()


# Definiendo la arquitectura de la red neuronal

# In[53]:


model.add(Dense(128, activation='relu', input_dim = 13))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')


# Entrenando la red neuronal

# In[54]:


model.fit(X_train, y_train, epochs=100)


# ### Evaluando los resultados

# In[55]:


y_pred = model.predict(X_test)


# In[56]:


from sklearn.metrics import r2_score


# In[57]:


r2 = r2_score(y_test, y_pred)


# In[58]:


print(r2)


# In[59]:


from sklearn.metrics import mean_squared_error
rmse = (np.sqrt(mean_squared_error(y_test, y_pred)))
print(rmse)

