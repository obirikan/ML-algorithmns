import sklearn
from sklearn import linear_model,preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv('car.data')
print(df.head())

#this is going to transfrom our non numerical value to numerical value
le=preprocessing.LabelEncoder()

#transforming texts to numerical 
buying=le.fit_transform(list(df['buying']))
maint=le.fit_transform(list(df['maint']))
door=le.fit_transform(list(df['door']))
persons=le.fit_transform(list(df['persons']))
lug_boot=le.fit_transform(list(df['lug_boot']))
safety=le.fit_transform(list(df['safety']))
cls=le.fit_transform(list(df['class']))

predict='class'
#grouping all
x=list(zip(buying,maint,door,persons,lug_boot,safety))
y=list(cls)

x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=0.1)



