#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
A = pd.read_csv("C:/Users/akaks/Downloads/Restaurant_Reviews.tsv",sep="\t")


# In[2]:


A.head()


# # Removing the special characters

# In[3]:


Q = []
from re import sub
for i in A.Review:
    Q.append(sub("[^a-zA-Z0-9 ]","",i.upper()))


# In[4]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


# In[22]:


#Q


# In[5]:


word_vect = cv.fit_transform(Q).toarray()


# In[6]:


word_vect


# In[7]:


words = cv.get_feature_names()


# In[23]:


#words


# # Spliting the data into training and testing set

# In[9]:


X = word_vect
Y = A.Liked


# In[10]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=21)


# In[11]:


X.shape


# # Creating a Neural Network

# In[12]:


from keras.models import Sequential
from keras.layers import Dense,Dropout
nn = Sequential()
nn.add(Dense(1000,input_dim=(2067)))
nn.add(Dropout(0.6))
nn.add(Dense(1000))
nn.add(Dropout(0.6))
nn.add(Dense(1,activation="sigmoid"))


# In[13]:


nn.compile(loss="binary_crossentropy",metrics="accuracy")
nn.fit(xtrain,ytrain,epochs=10,)


# # Predicting on testing set

# In[14]:


nn.predict(xtest)


# In[15]:


q=[]
for i in nn.predict(xtest):
    if (i[0]<0.5):
        q.append(0)
    else:
        q.append(1)
        
    


# In[16]:


from sklearn.metrics import accuracy_score
accuracy_score(ytest,q)


# # creating Function to classify the review

# In[20]:


def review_classification(str_):
    z = []
    z.append(sub("[^A-Za-z0-9 ]","",str_.upper()))
    x = cv.transform(z).toarray()
    pred = nn.predict(x)
    for i in pred:
        if i <0.5:
            print("Did Not Liked")
        else:
            print("Liked")


# In[21]:


review_classification("awesome")


# In[ ]:




