"""
Exercise on Multivariate Linear Regression
"""


import pandas as pd
import numpy as np
import math
from sklearn import linear_model
from word2number import w2n


dataFrame = pd.read_csv("/home/ubuntu/Documents/machine-learning/hiring.csv")

dataFrame.columns = dataFrame.columns.str.strip().str.lower()

dataFrame.experience = dataFrame.experience.fillna("zero")

dataFrame.experience = dataFrame.experience.apply(w2n.word_to_num)

avgOfTestScore = dataFrame["test_score(out of 10)"].median()

dataFrame["test_score(out of 10)"] = dataFrame["test_score(out of 10)"].fillna(avgOfTestScore)

newDataFrame = dataFrame.drop("salary($)", axis="columns")

model = linear_model.LinearRegression()
model.fit(newDataFrame[["experience", "test_score(out of 10)","interview_score(out of 10)"]], dataFrame["salary($)"])

experience = input("Enter the experience of employee: ")
test_score = input("Enter the test score of employee: ")
interview_score = input("Enter the interview score of employee: ")

predictedDataFrame = pd.DataFrame({"experience":[experience], "test_score(out of 10)":[test_score], "interview_score(out of 10)":[interview_score]})
predictionOfEmployeeSalary = model.predict(predictedDataFrame)

print("================================================")
print(f"The predicted salary for an employee with {experience} years of experience, a test score of {test_score} out of 10, and an interview score of {interview_score} out of 10 is: ${predictionOfEmployeeSalary[0]:.2f}")
print("================================================")



print("ML Model Run...")