import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('./loan_pred.csv')
#print(df)
#print(data.describe())

#print missing values in the dataset 
#print(data.apply(lambda d: sum(d.isnull()),axis=0 ))

#delete the rows with 0s
data.dropna(inplace = True)
#print(data.apply(lambda d: sum(d.isnull()),axis=0 ))

#list of categorical predictors
cat_var =['Gender','Married','Education', 'Self_Employed','Loan_Status' ]
le =LabelEncoder()

#transforming categorical predictors into numerical
for n in cat_var:
    data[n] = le.fit_transform(data[n])
    
data.dtypes
print(data.apply(lambda d: sum(d.isnull()),axis=0 ))

LoanAmount = data['LoanAmount'].values
Credit_History = data['Credit_History'].values
Loan_Status = data['Loan_Status'].values

fig= plt.figure()
ax = Axes3D(fig)
ax.scatter(LoanAmount,Credit_History,Loan_Status,color='#ef1234')
plt.show()
