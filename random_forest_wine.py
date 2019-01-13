#wine quality

#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
df=pd.read_csv("winequality-white.csv",sep=";")
print(df["quality"].unique())
print(df["quality"].describe())
print(df["quality"].value_counts())



def istasty(quality):
    if(quality>=7):
        return 2
    elif(quality>=5):
        return 1
    else:
        return 0
#in this we have make 3 categories of wine bad,average and good and assign values 0,1 and 2 to them respectively.
df["tasty"]=df["quality"].apply(istasty)
print(df.columns)

X=df.iloc[:,:-2].values
y=df.iloc[:,-1].values

#splitting the dataset into traininig and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#featuere scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#importing the clssifier here
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,random_state=0)
classifier.fit(X_train, y_train)
importances=classifier.feature_importances_
for i,features in zip(importances,['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                                             'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 
                                             'pH', 'sulphates', 'alcohol']):
    print("{}:{}".format(features,i))
indices = np.argsort(importances)

# Rearrange feature names so they match the sorted feature importances
names = [df.columns[i] for i in indices]

# Barplot: Add bars
plt.bar(range(X.shape[1]), importances[indices])
# Add feature names as x-axis labels
plt.xticks(range(X.shape[1]),names, rotation=40, fontsize = 8)
# Create plot title
plt.title("Feature Importance")
# Show plot
plt.show()



# Predicting the Test set results
y_pred= classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import zero_one_loss
print(zero_one_loss(y_test,y_pred,normalize=True))
#Zero-one classification loss.

'''If normalize is True, return the fraction 
of misclassifications (float), else it returns the number of misclassifications (int). The best performance is 0.'''

