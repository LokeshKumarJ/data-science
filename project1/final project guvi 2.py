#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[4]:


data = pd.read_csv("project data.csv")
data


# # DATA PREPROCESSING

# In[5]:


data["EducationField"].unique()


# In[6]:



data.replace({"Department":{"Sales":1,"Research & Development":2,"Human Resources":3}},inplace = True)


# In[7]:



data.replace({"BusinessTravel":{"Travel_Rarely":1,"Travel_Frequently":2,"Non-Travel":0}},inplace = True)


# In[8]:



data.replace({"EducationField":{"Life Sciences":1,"Medical":2,"Marketing":3,"Technical Degree":4,"Human Resources":5,"Other":6}},inplace = True)


# In[9]:



data.replace({"Gender":{"Male":0,"Female":1}},inplace = True)


# In[49]:



data.replace({"Attrition":{"Yes":0,"No":1}},inplace = True)


# In[176]:


data


# In[11]:


data.isnull().sum()


# In[12]:


data["NumCompaniesWorked"].unique()


# In[13]:


data = data.fillna(data.mean())


# In[20]:


data.style.set_precision(2)


# In[28]:


data["MaritalStatus"].unique()


# In[27]:


data["JobRole"].unique()


# In[26]:



data.replace({"JobRole":{"Healthcare Representative":1,"Research Scientist":2,"Sales Executive":3,"Human Resources":4,"Research Director":5,"Laboratory Technician":6,"Manufacturing Director":7,"Sales Representative":8,"Manager":9}},inplace = True)


# In[25]:



data.replace({"MaritalStatus":{"Married":0,"Single":1,"Divorced":2}},inplace = True)


# In[24]:


data.replace({"Over18":{"Y":1}},inplace = True)


# In[29]:


data.dtypes


# # ALGORITHM
#  # LOGISTIC REGRESSION

# In[30]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
x=data.drop(["Attrition","EmployeeID"],axis=1)
y=data["Attrition"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
predictions = logmodel.predict(x_test)


# In[31]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# In[32]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)


# In[33]:


from sklearn.metrics import classification_report
classification_report(y_test,predictions)


# In[168]:





# In[103]:





# In[194]:





# In[195]:





# In[129]:


x


# In[146]:





# In[144]:





# In[147]:


plt.boxplot(data["TotalWorkingYears"])


# In[198]:


plt.boxplot(data["DistanceFromHome"])


# In[149]:


plt.boxplot(data["TrainingTimesLastYear"])


# In[ ]:


plt.boxplot(data["TotalWorkingYears"])


# In[ ]:


plt.boxplot(data["TotalWorkingYears"])


# In[ ]:


plt.boxplot(data["TotalWorkingYears"])


# In[ ]:


plt.boxplot(data["TotalWorkingYears"])


# In[ ]:


plt.boxplot(data["TotalWorkingYears"])


# # RANDOM FOREST CLASSIFIER

# In[162]:


from sklearn.tree import DecisionTreeClassifier
treemodel = DecisionTreeClassifier(criterion="entropy")
treemodel.fit(x_train,y_train)
predict = treemodel.predict(x_test)


# In[163]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict)


# In[203]:



from sklearn.ensemble import RandomForestClassifier
random = RandomForestClassifier()
model = random.fit(x_train,y_train)
predicts = model.predict(x_test)


# In[204]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predict)


# # EDA

# In[34]:


data["Age"].plot(kind='hist',title = 'age')


# In[51]:


attrition = data[data["Attrition"] == 0]
attrition


# In[52]:


attrition["TotalWorkingYears"].plot(kind="hist")


# In[ ]:


#from this graph it shows that the workexp between 0 to 12 years has a more chance for attrition


# In[53]:


attrition["Age"].plot(kind="hist")


# In[ ]:


#the graph showing that the age group between 25 and 40 are more attrition


# In[ ]:





# In[ ]:




