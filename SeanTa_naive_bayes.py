#!/usr/bin/env python
# coding: utf-8

# In[87]:


#-------------------------------------------------------------------------
# AUTHOR: Sean Ta
# FILENAME: SeanTa_naive_bayes
# SPECIFICATION: Assignment#2 Question 5b
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv
#reading the training data in a csv file
#--> add your Python code here
dbTraining = []


#reading the data in a csv file
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTraining.append(row)
                
# transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
# X =

# transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> add your Python code here
# Y =

X = []
Y = []

outlook = {
    "Sunny": 1,
    "Overcast": 2,
    "Rain": 3}

temp = {
    "Hot": 1,
    "Mild": 2,
    "Cool": 3}

humidity = {
    "High": 1,
    "Normal": 2}

wind = {
    "Weak": 1,
    "Strong": 2}

playtennis = {
    "No" : 1,
    "Yes" : 2}

for data in dbTraining:
    X.append([outlook[data[1]], temp[data[2]], humidity[data[3]],wind[data[4]]])
    Y.append(playtennis[data[5]])

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTest = []
X_test = []

with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
             dbTest.append (row)

                
for data in dbTest:
    X_test.append([outlook[data[1]], temp[data[2]], humidity[data[3]],wind[data[4]]]) # transforming test data to numeric
    

#printing the header os the solution
print ("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
clf.predict_proba(X_test)
predictions = clf.predict(X_test)

results = []
playtennis = {
    1 : "No",
    2 : "Yes"}

for data in predictions:
    results.append(playtennis[data]) # transforming predictions as values of 'yes' or 'no'
    
for i,instance in enumerate(dbTest): # adding classifation confidence of predictions into each instance of test data
    dbTest[i].extend(clf.predict_proba(X_test)[i])
    
for i,instance in enumerate(dbTest): # printing instances with confidence levels greater than or equal to 0.75
    instance.insert(5,results[i]) #inserting predictions ('yes' or 'no') into each instance
    if instance[8] >= 0.75:  # if confidence of 'yes' is greater than 0.75
        print(str(instance[0]).ljust(15) + str(instance[1]).ljust(15) + str(instance[2]).ljust(15) + str(instance[3]).ljust(15) + str(instance[4]).ljust(15) + str(instance[5]).ljust(15) + str(instance[8]).ljust(15))
    if instance[7] >= 0.75: # if confidence of 'no' is greater than 0.75
        print(str(instance[0]).ljust(15) + str(instance[1]).ljust(15) + str(instance[2]).ljust(15) + str(instance[3]).ljust(15) + str(instance[4]).ljust(15) + str(instance[5]).ljust(15) + str(instance[7]).ljust(15))

