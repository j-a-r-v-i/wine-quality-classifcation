#wine quality

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#importing the dataset
dataset=pd.read_csv("winequality-white.csv",sep=";")
def istasty(quality):
    if(quality>=7):
        return 2
    elif(quality>=5):
        return 1
    else:
        return 0
dataset["tasty"]=dataset["quality"].apply(istasty)
X=dataset.iloc[:,:-2].values
y=dataset.iloc[:,-1].values

#splitting the dataset into traininig and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#featuere scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#importing the clssifier here
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
print(classifier.score(X_train, y_train))

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
sklearn.metrics.mean_squared_error(y_test,y_pred)
