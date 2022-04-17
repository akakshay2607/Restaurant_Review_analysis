#Reading data
import pandas as pd
A = pd.read_csv("C:/Users/akaks/Downloads/Restaurant_Reviews.tsv",sep="\t")
A.head()

#Removing Special Characters
Q = []
from re import sub
for i in A.Review:
    Q.append(sub("[^a-zA-Z0-9 ]","",i.upper()))
    
#Converting data into vectorised format
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

word_vect = cv.fit_transform(Q).toarray()

#Getting all the words
words = cv.get_feature_names()

#Defining X and Y
X = word_vect
Y = A.Liked

#Spliting data in training and testing set
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.2,random_state=21)

X.shape

#Creating Neural Network
from keras.models import Sequential
from keras.layers import Dense,Dropout
nn = Sequential()
nn.add(Dense(1000,input_dim=(2067)))
nn.add(Dropout(0.6))
nn.add(Dense(1000))
nn.add(Dropout(0.6))
nn.add(Dense(1,activation="sigmoid"))
nn.compile(loss="binary_crossentropy",metrics="accuracy")
nn.fit(xtrain,ytrain,epochs=10,)

#predicting on testing data
nn.predict(xtest)

q=[]
for i in nn.predict(xtest):
    if (i[0]<0.5):
        q.append(0)
    else:
        q.append(1)

#Checking The accuracy of model
from sklearn.metrics import accuracy_score
accuracy_score(ytest,q)

#Defining Function for Review Classification
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

review_classification("awesome")
