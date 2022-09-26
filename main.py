import tensorflow
import keras
import pandas as pd
from matplotlib import pyplot
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# import Data
data = pd.read_csv("student-mat.csv", sep=";")

#take out the wanted labels / use wanted independent variable
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#dependent variable
predict = "G3"

#independent variable(s)
x = np.array(data.drop([predict], 1))

#dependent variable
y = np.array(data[predict])

#testing and training using actual data to know correlation btn them
#test_size is the percentage used for testing the remaining goes to training 
x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.5)
#use the linearRegression module
linear=linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc=linear.score(x_test,y_test)
print(acc)
print('co: \n',linear.coef_)
print('intercept: \n',linear.intercept_)
v=[12 ,14 , 1 , 0 , 0]
predictions=linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])


