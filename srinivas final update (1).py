#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sl
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pydotplus
import seaborn as sns; sns.set_theme()
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier


# In[2]:


phish=pd.read_csv('phishing_url_dataset.csv')


# In[3]:


phish


# In[4]:


phish.shape


# In[5]:


#Data Pre-processing, Deleting duplicate data in the dataset
Dup= phish.drop_duplicates
Dup


# In[6]:


#Finding the null values in the dataset
phish.isnull()


# In[7]:


#feature selection from the above dataset 
selected_cols=['url_length','valid_url','sensitive_words_count','at_symbol','path_length','nb_com']


# In[8]:


#Data preparation
x=phish[selected_cols]
x
y=phish['target']
y=np.array(y)


# In[9]:


#splitting the dataset
#By splitting the testing and training data the ratio of the testing data is 30% and training data is 70%
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size=0.3, random_state=1)


# In[10]:


print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[11]:


#RandomForest classifier
RFC=RandomForestClassifier(n_estimators=100,random_state=100)
classifier=RFC.fit(X_train,y_train)
prediction=classifier.predict(X_test)


# In[12]:


print("The accuracy of the RandomForest classifier:", metrics.accuracy_score(y_test,prediction))


# In[13]:


#Classification Report of the Random forest classifier
print(classification_report(y_test,prediction))


# In[14]:


#Model evaluation ,metrics confusion matrix of Random forest classifier
cf_matrix1 = confusion_matrix(y_test,prediction)
cf_matrix1


# In[15]:


#heatmap of the Random forest , 
RFC = sns.heatmap(cf_matrix1/np.sum(cf_matrix1), annot=True, fmt='.2%', cmap='Blues')
RFC.set_title ('RandomForest confusion matrix with labels\n\n');
RFC.set_xlabel('\nPredicted Values')
RFC.set_ylabel('Actual Values')
RFC.xaxis.set_ticklabels(['False','True'])
RFC.yaxis.set_ticklabels(['False','True'])
#Visualisation of the Random forest Confusion matrix plot
plt.show()


# In[16]:


#ROC CURVE FOR Random Forest
fpr, tpr, threshold = metrics.roc_curve(y_test, prediction)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
plt.title('Receiver Operating Characteristic for RandomForest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[18]:


#SVM Classifier
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)
svm_classifier=svm.SVC(kernel='linear')
s=svm_classifier.fit(X_train,y_train)
y_prediction=s.predict(X_test)


# In[19]:


print("The accuracy of the SVM classifier:", metrics.accuracy_score(y_test,y_prediction))


# In[20]:


#Classification Report of the SVM
print(classification_report(y_test,y_prediction))


# In[21]:


#Model evaluation ,metrics confusion matrix of SVM
cf_matrix2 = confusion_matrix(y_test,y_prediction)
cf_matrix2


# In[22]:


#heatmap of the SVM 
RFC = sns.heatmap(cf_matrix2/np.sum(cf_matrix2), annot=True, fmt='.2%', cmap='Blues')
RFC.set_title ('SVM with labels\n\n');
RFC.set_xlabel('\nPredicted Values')
RFC.set_ylabel('Actual Values')
RFC.xaxis.set_ticklabels(['False','True'])
RFC.yaxis.set_ticklabels(['False','True'])
#Visualisation of the SVM Confusion matrix plot
plt.show()


# In[23]:


#ROC CURVE for SVM
fpr, tpr, threshold = metrics.roc_curve(y_test, y_prediction)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
plt.title('Receiver Operating Characteristic for RandomForest')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[24]:


#DecisionTreeClassifier
clf=DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
Decision_pre= clf.predict(X_test)
accuracyscore= clf.score(X_test,y_test)
print(accuracyscore)


# In[25]:


#Classification Report of the Decisiontree
print(classification_report(y_test,Decision_pre))


# In[26]:


#Model evaluation ,metrics confusion matrix of Decisiontree
cf_matrix3 = confusion_matrix(y_test,Decision_pre)
cf_matrix3


# In[27]:


#heatmap of the Decisiontree 
RFC = sns.heatmap(cf_matrix3/np.sum(cf_matrix3), annot=True, fmt='.2%', cmap='Blues')
RFC.set_title ('Decisiontree confusion matrix with labels\n\n');
RFC.set_xlabel('\nPredicted Values')
RFC.set_ylabel('Actual Values')
RFC.xaxis.set_ticklabels(['False','True'])
RFC.yaxis.set_ticklabels(['False','True'])
#Visualisation of the Decisiontree Confusion matrix plot
plt.show()


# In[28]:


#ROC CURVE for Decisiontree
fpr, tpr, threshold = metrics.roc_curve(y_test, Decision_pre)
roc_auc = metrics.auc(fpr, tpr)
plt.figure()
plt.title('Receiver Operating Characteristic for Decisiontree')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




