# import tensorflow
# import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle


#import Data using csv
data = pd.read_csv("student-mat.csv", sep=";")


#take out the wanted labels / use wanted independent variable
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#dependent variable (Y)
predict = "G3"

#independent variable(s)...(X)
x = np.array(data.drop([predict],1))

#dependent variable
y = np.array(data[predict])
print(y)
print(x)
#training and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.09)
'''''''''
best=0
#testing and training using actual data to know correlation btn them
#test_size is the percentage used for testing the remaining goes to training
for _ in range(10000):
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.09)

    #use the linearRegression module
    linear=linear_model.LinearRegression()

    #minimizing error(gradient decent)
    linear.fit(x_train,y_train)

    #test the accuracy of the error
    acc=linear.score(x_test,y_test)
    print(acc)

    #printing out some values for verification
    #test model accuracy level


    if acc>best:
        best=acc
        with open('studentmodel.pickle','wb') as f:
           pickle.dump(linear,f)
'''''''''

savedmodel=open('studentmodel.pickle', 'rb')
newlinear=pickle.load(savedmodel)

#using my own data to test module
v=[[1 ,1 , 0 ,3, 5]]

#predict the outcome of your value(s)
predictions=newlinear.predict(v)

#loop through prediction to see if your data is corresponding well
for x in range(len(predictions)):
    print(predictions[x])


#plot your data
# plt.scatter(data['G1'],data['G3'])
# plt.xlabel('G1')
# plt.ylabel('final grades')
# plt.show()
