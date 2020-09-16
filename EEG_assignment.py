#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
import tensorflow as tf


# In[2]:

print('now start, wait....')
#file_path = 'E:\Coding\Machine learning & Deep learning code\EEG files'
obs = pd.read_pickle('Data_Spectrograms.pkl')
#data_raw=pd.read_pickle(file_path+'\Data_Raw_signals.pkl')
test = pd.read_pickle('Test_Spectrograms_no_labels.pkl')


# In[3]:


batch_size = 128
data = obs[0]
label = tf.keras.utils.to_categorical(obs[1])


# In[5]:


data_demo = np.reshape(data,(15375,-1,30))


# In[8]:


X_train, X_val, y_train, y_val = train_test_split(data_demo, label, test_size=0.30,random_state=44)
y_val.shape


# In[9]:


train_set = tf.data.Dataset.from_tensor_slices((X_train,y_train))
val_set = tf.data.Dataset.from_tensor_slices((X_val,y_val))


# In[10]:


train_set = train_set.repeat(3).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_set = val_set.repeat(3).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


# In[11]:


def build_model(hp):
    # define the hyper parameter ranges for thelearning rate, dropout and hidden unit
    hp_units_1 = hp.Int('filters', min_value=16,max_value=64, step=16)
    hp_units_3 = hp.Int('units', min_value=6,max_value=10, step=2)
    hp_units_4 = hp.Int('filters_1', min_value=16,max_value=64, step=16)
    hp_dropout = hp.Choice('dropout',values=[0.3,0.4,0.5])
    hp_dropout_1 = hp.Choice('dropout_1',values=[0.3,0.4,0.5])
    lr = hp.Choice('lr',values=[1e-3,2e-3,1e-4,2e-4])
    hp_layer_1 = hp.Int('layer_1', min_value=0, max_value=1, step=1)
    hp_layer_2 = hp.Int('layer_2', min_value=0, max_value=1, step=1)
    
    input_tensor = tf.keras.layers.Input(shape=(200,30))
    x = tf.keras.layers.Conv1D(hp_units_1,  5, input_shape=(200, 30),activation='relu')(input_tensor)
    p = tf.keras.layers.MaxPooling1D(3)(x)
    p = tf.keras.layers.Dropout(hp_dropout)(p)
    for layer in range(hp_layer_1):
        p = tf.keras.layers.Conv1D(hp_units_1,  5, activation='relu')(p)
        p = tf.keras.layers.MaxPooling1D(3)(p)
    
    for layer in range(hp_layer_2):
        p = tf.keras.layers.GRU(units=hp_units_4,return_sequences=True)(p)
        
    p = tf.keras.layers.GRU(units=hp_units_4)(p)
    p = tf.keras.layers.Dropout(hp_dropout_1)(p)
    p = tf.keras.layers.Dense(hp_units_3, activation='relu')(p)
        
    #p = tf.keras.layers.Dropout(hp_dropout)(p)
    p = tf.keras.layers.BatchNormalization()(p)
    
    output_tensor = tf.keras.layers.Dense(6, activation='softmax')(p)
    model = tf.keras.Model(input_tensor,output_tensor)
    model.compile(optimizer = tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics = ['accuracy'])
    
    return model


# In[12]:


tuner = RandomSearch(
    build_model,
    objective='val_accuracy', 
    max_trials=40,
    executions_per_trial=2,
    directory='random12') 


# In[13]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,restore_best_weights=True)


# In[14]:


tuner.search(train_set,epochs=80, validation_data=(val_set),callbacks=[early_stopping],verbose=2)


# In[15]:


tuner.results_summary()


# In[16]:


model = tuner.get_best_models(num_models=1)


# In[17]:


test_loss, test_acc = model[0].evaluate(val_set)


# In[18]:


test_acc


# In[19]:


test_data = np.array(test)


# In[20]:


test_data = np.squeeze(test_data, axis=0)


# In[21]:


test = np.reshape(test_data,(test_data.shape[0],-1,test_data.shape[-1]))


# In[22]:


result = model[0].predict(test)


# In[23]:


results = [np.argmax(result[i]) for i in range(len(result))]


# In[24]:


df = pd.DataFrame({"label": results})


# In[25]:


df.to_csv('answer.txt', header=None, index=False)

