import pandas as pd

print('introduction to pandas')
PATH = './data/athlete_events.csv'

#-----reading the file
df = pd.read_csv(PATH)
df= df.head(20)
#-----first 20 and last 20------------------------------------------
#print(df.head(20))#first 20
#print(df.tail(20))#last 20

#-----displaying the data---------------------------------------------
#print(df.dtypes)#- shows all the column attributes and their data types

#print(df.index)#shows number of rows(0 - 271116)

#print(df.columns)#- shows all the columns

#print(df.values)#- shows all the rows

#print(df.describe)#statistical summary of the data

#print(df.sort_values('Age',ascending='True')) # sort values by age

#--------slicing records---------------
# print(df.Age)
# print(df['Age'])

#print(df[2:20])#prints rows 2 to 19

#print( df[['Age', 'Sex']] ) #displays the columns in the brackets(notice the 2 square braces)

#print( df.loc[ 1 ,['Age'] ] )#displays the age of the row in index 1

#print(df)
#print( df.iloc[ 3:15 ,[0,4] ] )#displays the rows(3-15) of the datain column 0 and 4


#-------filtering data--------
#print(df.Age > 30)#prints a boolean showing all rows with ages above 30

print(df)

print( df[ df['City'].isin(['Albertville','Lillehammer'])])



# 1. How old were the youngest male and female participants of the 1996 Olympics?

