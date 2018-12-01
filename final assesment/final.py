import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D


#-------read csv data
data = pd.read_csv('./California_cities.csv')


#-------check for missing values
#print(data.apply(lambda d: sum(d.isnull()),axis=0 ))



#-------------delete the rows with 0s
data.dropna(inplace = True)
#print(data.apply(lambda d: sum(d.isnull()),axis=0 )) 

#------------mean of last 10 latitude values
#print(" mean ", data[["latd"]].tail(10).mean() )

#------------max of last 10 latitude values
#print(" max ", data[["latd"]].tail(10).max())

#------------min of last 10 latitude values
#print(" min ",data[["latd"]].tail(10).min())

#------------standard deviation of last 10 latitude values
#print(" standard deviation ",data[["latd"]].tail(10).std())

#-------------median of last 10 latitude values
#print(" median ",data[["latd"]].tail(10).median())

#-------------display the 10 cities with the least elevation in metres
#result = data.sort_values('elevation_m',axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#print(result[['city']].head(10))

#-------------top ten cities with the highest population totals
#result2 = data.sort_values('population_total',axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last')
#print(result2[['city']].head(10))


#------------Plot the relationship of the top ten highest areas in feet with their respective population totals
#result3 = data.sort_values('elevation_ft',axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').head(10)

#plt.scatter(result3['elevation_ft'], result3['population_total'])
#plt.show()

#-------------Plot a histogram of the area_total_sq_mi and discribe the patter of the data (you can use 20 bins for the histogram plot)
# n_bins = 20
# x = data[[ 'area_total_sq_mi' ]]
# y = data[[ 'city' ]]

# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)


# axs[0].hist(x, bins=n_bins)
# axs[1].hist(y, bins=n_bins)

# plt.show()

#-------------Normalize the values of the area_total_sq_mi, plot a histogram of the values and describe the pattern of the data after normalization.

# result4 = data['area_total_sq_mi']

# norm = [float(i)/sum(result4) for i in result4]

# n_bins = 20
# x = norm
# y = data[[ 'city' ]]

# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)


# axs[0].hist(x, bins=n_bins)
# axs[1].hist(y, bins=n_bins)

# plt.show()

#-------------Using a plot analyze the relationship of the population total by the area total in sq mi
# x = data['population_total']
# y = data['area_total_sq_mi']

# plt.plot(x, y)
# plt.show()

#------------Assuming population_total to be our target value, fit the data onto a linear regression model and evaluate the performance of the model	
#list of categorical predictors
cat_var =['city','latd','longd', 'elevation_m','elevation_ft','area_total_sq_mi','area_land_sq_mi','area_water_sq_mi','area_total_km2','area_land_km2','area_water_km2','area_water_percent' ]
le =LabelEncoder()

#A for loop to transform the categorical values to numerical values
for n in cat_var:
    data[n] = le.fit_transform(data[n])

#Getting the variables to an array.
Total_area_in_sqm = data['area_total_sq_mi'].values
water_area_percnt = data['area_water_percent'].values
elevation = data['elevation_m'].values

# Plotting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Total_area_in_sqm, water_area_percnt, elevation, color='#ef1234')
#plt.show()

#Now we generate our parameters(the theta values)
m = len(Total_area_in_sqm)
x0 = np.ones(m)
X = np.array([x0, Total_area_in_sqm, water_area_percnt]).T
# Initial Coefficients
B = np.array([0, 0, 0])
Y = np.array(elevation)
alpha = 0.0001

#We’ll define our cost function.
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, Y, B)
print("Initial Cost")
print(inital_cost)

#Defining the Gradient Descent
def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

# 100 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100)

# New Values of B
print("New Coefficients")
print(newB)

# Final Cost of new B
print("Final Cost")
print(cost_history[-1])

# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = X.dot(newB)

print("RMSE")
print(rmse(Y, Y_pred))
print("R2 Score")
print(r2_score(Y, Y_pred))




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# X and Y Values
X = np.array([Total_area_in_sqm, water_area_percnt]).T
Y = np.array(elevation)

# Model Intialization
reg = LinearRegression()
# Data Fitting
reg = reg.fit(X, Y)
# Y Prediction
Y_pred = reg.predict(X)

# Model Evaluation
rmse = np.sqrt(mean_squared_error(Y, Y_pred))
r2 = reg.score(X, Y)

print(rmse)
print(r2)