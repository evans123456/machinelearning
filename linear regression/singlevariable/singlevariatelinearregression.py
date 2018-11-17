import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv("./lifeexpectancy_usa.csv")

#print(dataset.describe())

#need to find the dependent and independent variables to get the x and y coordinates
x = dataset['TIME'].values #getting the values of x
#print(x)#print values of x coordinate ---> time in years

y = dataset['Value'].values #getting the values of y
#print(y)#print values of y coordinate ---> life expectancy value

#to get the coeficient to use as .... i.e finding the values of 0x and 0y(needed for the calculation of parameter values)
mean_x = np.mean(x)
#print(mean_x)

mean_y = np.mean(y)
#print(mean_y)

m = len(x)
number = 0
denominator  = 0

#calculates the values of the coeffitients i.e 00 and 01
for i in range(m):
    number += (x[i] - mean_x)*(y[i]-mean_y)
    denominator += (x[i] - mean_x)**2
    b1= number / denominator
    b0 = mean_y - (b1*mean_x)
    
print(b1,b0)

#representing our model graphically 
max_x = np.max(x) + 10
min_x = np.min(x) - 10

x = np.linspace(min_x,max_x,10)
y = b0 + b1 * x

plt.plot(x,y,color='black',label='regression-line') 

plt.scatter(x,y,c='blue',label='scatter-plot')

plt.xlabel('period in years')
plt.ylabel('value in years')
plt.legend()
plt.show()

#(NEXT CLASS)multivariant linear regression
ss_t = 0
ss_r = 0
for i in range(m):
    y_pred = b0 + b1 * x[i]
    ss_t +=(y[i]-mean_y)**2
    ss_r +=(y[i] - y_pred)**2

r2 = 1 - (ss_r/ss_t)
print(r2)
    
