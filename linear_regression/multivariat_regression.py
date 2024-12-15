"""
Multiple Variable Linear Regression

Or

Multivariate Linear Regression
"""


import pandas as pd
import numpy as np
from sklearn import linear_model


dataFrame = pd.read_csv("/home/ubuntu/Documents/machine-learning/homeprices.csv")

dataFrame.columns = dataFrame.columns.str.strip().str.lower()

avgOfBedrooms = dataFrame.bedrooms.median()

dataFrame.bedrooms = dataFrame.bedrooms.fillna(avgOfBedrooms)

newDataFrame = dataFrame.drop("price", axis="columns")

model = linear_model.LinearRegression()
model.fit(newDataFrame[["area", "bedrooms", "age"]], dataFrame["price"])

area = input("Enter Predicted Home Area: ")
bedrooms = input("Enter Predicted Home Bedrooms: ")
age = input("Enter Predicted Home Age: ")

print(f"Predicted Area(sq. ft.): {area} | Bedrooms: {bedrooms} | Age: {age}")

predictionDataFrame = pd.DataFrame({"area":[area], "bedrooms":[bedrooms], "age":[age]})
predictedHomePrice = model.predict(predictionDataFrame)

print("================================================")
print(f"The predicted price for a home with an area of {area} sq. ft, {bedrooms} bedrooms, and {age} years old is ${predictedHomePrice[0]:.2f}.")
print("================================================")


print("ML Model Run...")