"""
Created on Tue Jan 11 08:54:57 2022

@author: Darshan
"""

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


df = pd.read_csv('CancerPatients.csv')
print(df.isnull().sum())

df1 = df.drop(labels=['Patient Id'],axis=1,inplace=False)
Patient_Id = df['Patient Id']
col = df1.columns


print(col)

lst = list(col)
c=0
ls = [3,4,8,9,11,14,15,17]

for i in ls:
    lst.pop(i-c)
    c+=1
    

for i in lst:
    figure = plt.figure(figsize=(15,10))
    sns.countplot( x=i,hue='Gender',data=df1)
    plt.show()
 
    
df1.plot()
plt.show()

#Correlation
data_x = df1.drop(labels=['Level'], axis=1, inplace=False)
data_y = df1['Level']

corr_Pearson = df.corr(method='pearson')

figure = plt.figure(figsize=(15,10))
sns.heatmap(corr_Pearson,vmin=-1,vmax=+1,cmap='Blues',annot=True, 
            linewidths=1,linecolor = 'white')
plt.title('Pearson Correlation')
plt.show()


#Prediction --- Confusion Matrix
x_train, x_test, y_train, y_test = train_test_split(data_x,data_y,test_size = 0.3, random_state = 44,shuffle = True)

model = LogisticRegression(solver="liblinear").fit(x_train,y_train)
predict = model.predict(x_test)
cf_matrix = confusion_matrix(y_test,predict)
figure = plt.figure(figsize=(20,10))
sns.heatmap(cf_matrix, linecolor = "white", linewidth = 1,cmap = "Blues", annot = True)
plt.show()
