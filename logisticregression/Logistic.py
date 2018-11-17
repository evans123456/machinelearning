#===================================Exploring Trip History Dataset==============================================
#Import the relevant libraries
import pandas as pd
import numpy as np                     # For mathematical calculations               
import matplotlib.pyplot as plt        # For plotting graphs
#%matplotlib inline
import seaborn as sns # For data visualization
sns.set_style('whitegrid')
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Load data onto memmory using the pandas library
data = pd.read_csv("./sub_data.csv")

data.head()

#Lets analyse the measures of central dispersion of the data for the few columns with Numeric values

data.describe()
#Check the variables/features of the dataset
data.columns

#Check the first five rows of the datafrmae
data.head()

#Univariate Analysis
#Exploring the Target Variable. Our target variable is the Member Type column
data['Member Type'].value_counts()


#Checking for missing values in any column/features
data.isnull().sum()


#Convert the Categorical Values to Numerical to allow us perform plotting
#import the library LabelEncoder
from sklearn.preprocessing import LabelEncoder
#Create a list with categorical predictors
cat_var =['Start station','End station','Bike Number','Member Type']
#Initiate LabelEncoder
le = LabelEncoder() 
#A for loop to transform the categorical values to numerical values
for n in cat_var:
    data[n] = le.fit_transform(data[n])

#Checking for the type of the predictors afterwards
data.dtypes


#Explore the relationship between duration and member type
data.plot( x='Duration', y='Member Type',style='*')  
plt.title('Duration of Bike Use')  
plt.xlabel('Duration')  
plt.ylabel('Member Type')  
plt.show() 

#Explore the relationship between Start Station and member type
data.plot( x='Start station', y='Member Type',style='*')  
plt.title('Start station by Member Type')  
plt.xlabel('Start station')  
plt.ylabel('Member Type')  
plt.show()  


#Explore the relationship between End Station and member type
data.plot( x='End station', y='Member Type',style='*')  
plt.title('End station by Member Type')  
plt.xlabel('End station')  
plt.ylabel('Member Type')  
plt.show()  

#Explore the relationship between Bike Number and member type
data.plot( x='Bike Number', y='Member Type',style='*')  
plt.title('Bike Number by Member Type')  
plt.xlabel('Bike Number')  
plt.ylabel('Member Type')  
plt.show()  

#Explore the relationship between End Station and Duration
data.plot( x='End station', y='Duration',style='*')  
plt.title('End station by Duration')  
plt.xlabel('End station')  
plt.ylabel('Duration')  
plt.show()  


#Explore the relationship between Start Station and Duration
data.plot( x='Start station', y='Duration',style='*')  
plt.title('Start station by Duration')  
plt.xlabel('Start station')  
plt.ylabel('Duration')  
plt.show() 

#Lets plot a simple bar graph on the target variable
data['Member Type'].value_counts().hist()

#Lets plot a simple bar graph on the target variable and confirm the output we got earlier
data['Member Type'].value_counts().plot.bar();

#We can take a look at Member Type  by Duration

data.boxplot(column='Member Type', by = 'Duration')
plt.suptitle("");

data.boxplot(column='Member Type', by = 'Start station')
plt.suptitle("");

#Analysing more than one variable
#We can take a look at Member Type by End Station
data.boxplot(column='Member Type', by = 'End station')
plt.suptitle("")

#Analysing more than one variable
#We can take a look at Member Type by Start Station
data.boxplot(column='Member Type', by = 'Start station')
plt.suptitle("")


#Creation of Duration categories and analysing them per Member Type
bins=[0,2500,4000,6000,25000]
group=['Low','Average','High', 'Very high']
data['Duration']=pd.cut(data['Duration'],bins,labels=group)

Duration=pd.crosstab(data['Duration'], data['Member Type'])
Duration.div(Duration.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Duration')
P = plt.ylabel('Member Type')

#Creation of Duration categories and analysing them per Start Station
bins=[0,2500,4000,6000,25000]
group=['Low','Average','High', 'Very high']
data['Duration']=pd.cut(data['Duration'],bins,labels=group)

Duration=pd.crosstab(data['Duration'], data['Start station'])
Duration.div(Duration.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Duration')
P = plt.ylabel('Start station')


#Creation of Duration categories and analysing them per End Station
bins=[0,2500,4000,6000,25000]
group=['Low','Average','High', 'Very high']
data['Duration']=pd.cut(data['Duration'],bins,labels=group)

Duration=pd.crosstab(data['Duration'], data['End station'])
Duration.div(Duration.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Duration')
P = plt.ylabel('End station')

#Creation of Duration categories and analysing them per Bike Number
bins=[0,2500,4000,6000,25000]
group=['Low','Average','High', 'Very high']
data['Duration']=pd.cut(data['Duration'],bins,labels=group)

Duration=pd.crosstab(data['Duration'], data['Bike Number'])
Duration.div(Duration.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Duration')
P = plt.ylabel('Bike Number')

data.plot(x='Duration', y='Member Type', style='*')  
plt.title('Member Type by Duration')  
plt.xlabel('Duration')  
plt.ylabel('Member Type')  
plt.show() 


#import the library LabelEncoder
from sklearn.preprocessing import LabelEncoder
#Create a list with categorical predictors
cat_var =['Start station','End station','Bike Number','Member Type']
#Initiate LabelEncoder
le = LabelEncoder() 
#A for loop to transform the categorical values to numerical values
for n in cat_var:
    data[n] = le.fit_transform(data[n])

#Checking for the type of the predictors afterwards
data.dtypes

#Creating variables x is ithe input and Y is the target
X = data.iloc[:, 0:6]

Y = data[data.columns[-1]]


data.head()


#The Model using sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

log_reg = LogisticRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=3)

log_reg.fit(X_train, Y_train)

from sklearn import metrics

Y_pred = log_reg.predict(X_test)

print(metrics.accuracy_score(Y_test, Y_pred))

#===================================The End==============================================
