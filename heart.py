#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

#Removes a warning caused by Logistic Regression
import warnings
warnings.filterwarnings('ignore')


#--------------Load Dataset and Find Correlations--------------#


#Loading the dataset
heart = pd.read_csv('heart.csv')

X = heart.iloc[:,:-1].values
y = heart["target"]   

print("\n-Sample Data-")
print(heart.sample(10))

#Heat Map Correlation
cor = heart.corr().abs()
plt.figure(figsize=(12,10))
fig1 = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)


#Selecting highly correlated features
cor_target = abs(cor["target"])
print("\n-Feature Correlations to Target-")
print(cor_target[cor_target>-1].sort_values(ascending=False))


#--------------Testing/Training before Dropping Features--------------#


#Test/Train Before Dropping Features
print("\n\n---Test/Train Before Dropping Features---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("\n-Train-")
print(X_train.shape, y_train.shape)
print("\n-Test-")
print(X_test.shape, y_test.shape)


#Logistic Regression
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print("\n-Predictions-")
print(predictions)


#Flatten Values 
predict = predictions.flatten()
actual = y_test.values.flatten()


#Predictions vs Actual
df = pd.DataFrame({'Actual': actual, 'Predicted': predict})
print("\n-Acutal vs Predicted-")
with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
    print(df)


#Accuracy Score
score = round(accuracy_score(predictions,y_test)*100,2)
print("\n-Accuracy Score-")
print("Score: " + str(score) + "%")


#Confusion Matrix before Feature Drop
cf_matrix = confusion_matrix(actual, predict)
plt.figure(figsize=(8,6))
fig2 = sns.heatmap(cf_matrix, annot=True, yticklabels=[1,0], xticklabels=[1,0])
fig2.set(xlabel="Actual", ylabel="Predicted")
fig2.xaxis.tick_top()
fig2.xaxis.set_label_position('top')


#--------------Dropping Less Correlated Features and Test/Train--------------#


#Dropping Less Correlated Features
heart.drop(['fbs', 'chol', 'restecg', 'trestbps', 'age', 'sex', 'thal', 'slope', 'ca'], axis=1, inplace=True)
X = heart.iloc[:,:-1].values  
print("\n-Sample after Dropping less Correlated Features- ")
print(heart.sample(10))


#Test/Train After Dropping Features
print("\n\n---Test/Train After Dropping Features---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("\n-Train-")
print(X_train.shape, y_train.shape) #Train
print("\n-Test-")
print(X_test.shape, y_test.shape) #Test


#Logistic Regression
lr = LogisticRegression()
model = lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print("\n-Predictions-")
print(predictions)


#Flatten Values 
predict = predictions.flatten()
actual = y_test.values.flatten()


#Predictions vs Actual
df = pd.DataFrame({'Actual': actual, 'Predicted': predict})
print("\n-Acutal vs Predicted-")
with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
    print(df)


#Accuracy Score
score = round(accuracy_score(predictions,y_test)*100,2)
print("\n-Accuracy Score-")
print("Score: " + str(score) + "%")


#Confusion Matrix After Feature Drop
cf_matrix = confusion_matrix(predict, actual)
plt.figure(figsize=(8,6))
fig3 = sns.heatmap(cf_matrix, annot=True, yticklabels=[1,0], xticklabels=[1,0])
fig3.set(xlabel="Actual", ylabel="Predicted")
fig3.xaxis.tick_top()
fig3.xaxis.set_label_position('top')


#Show Figures
print(plt.show())


#--------------Input Own Values and Make Prediction--------------#


#New Predictions
print("\n\n---New Prediction---")


#Change Features to test Predictions
cp = 3 #0-3
thalach = 180 #71-202
exang = 1 #1=yes, 0=no
oldpeak = 1 #0-6.2


Xnew = [[cp, thalach, exang, oldpeak]]
ynew = model.predict(Xnew)


#Change Features to test Predictions
print("cp: " + str(cp))
print("thalach: " + str(thalach))
print("exang: " + str(exang)) 
print("oldpeak: " + str(oldpeak))
print("---------------")
print("Target: " + str(ynew[0])) #1=Yes, 0=No

