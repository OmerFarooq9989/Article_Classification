#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold


# In[2]:


train_dir = "C://Users//omerf//Desktop//data//train"
test_dir = "C://Users//omerf//Desktop//data//test"

count1 = -1
val =[]
y_train = []
labelss = []

for i in os.listdir(train_dir):
    count1 += 1
    labelss.append(i)
    temp = train_dir+"//"+i
    for j in os.listdir(temp):
        temp1 = temp+"//"+j
        with open(temp1, 'r') as f:
            data = ""
            data += f.read()
            data = data.lower()
            val.append(data)
            y_train.append(count1)
print("Read Training directory")

count2 = -1
y_test = []
for i in os.listdir(test_dir):
    count2 += 1
    temp = test_dir+"//"+i
    for j in os.listdir(temp):
        temp1 = temp+"//"+j
        with open(temp1, 'r') as f:
            data = ""
            data += f.read()
            data = data.lower()
            val.append(data)
            y_test.append(count2)
print("Read testing directory")
    



# In[3]:



vectorizer = CountVectorizer(analyzer='word',min_df = 5,max_df = 10000)
data = vectorizer.fit_transform(val)
print(vectorizer.get_feature_names())



# In[4]:



print(len(vectorizer.get_feature_names()))
y_train1 = np.array(y_train)
y_test1 = np.array(y_test)
print(type(y_train1),len(y_test))

data_conv = data.toarray()
print(data_conv.shape)
train_data = data_conv[:len(y_train)]
test_data = data_conv[len(y_train):]

print((train_data.shape),(test_data.shape))

input_s = train_data.shape[1]
scores = []
epochs = 10

from sklearn.svm import LinearSVC
model = LinearSVC(tol=1.0e-4,max_iter=5000,verbose=3)
model.fit(train_data, y_train1)
predicted_labels = model.predict(test_data)

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
print (confusion_matrix(y_test, predicted_labels))
print (classification_report(y_test, predicted_labels, digits=4))
print ("Accuracy of the model using SVM is:" + str((accuracy_score(y_test, predicted_labels))*100))

h =  (classification_report(y_test, predicted_labels, digits=4))

for ind,val in enumerate(labelss):
    print(str(ind) +" : "+ val)
