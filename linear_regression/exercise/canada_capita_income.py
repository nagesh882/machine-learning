"""
Exercise on Simple Linear Regression
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



dataFrame = pd.read_csv("/home/ubuntu/Documents/machine-learning/canada_per_capita_income.csv")

dataFrame.columns = dataFrame.columns.str.strip().str.lower()

x = dataFrame["year"]
y = dataFrame["per capita income (us$)"]

m, b = np.polyfit(x, y, 1)

plt.figure(figsize=(8, 6))

plt.scatter(x, y, color="red", marker="o", label="Data Points")
plt.plot(x, x * m + b, color="blue", label="Regression Line")
plt.plot(x, y, color="magenta", label="Regression Line")

plt.xlabel("Year")
plt.ylabel("Per Capita Income(USD$)")
plt.title("Per Capita Income vs. Year")

plt.legend()
plt.savefig("canada_per_capita_income.jpg")

newDataFrame = dataFrame.drop("per capita income (us$)", axis="columns")

model = linear_model.LinearRegression()
model.fit(newDataFrame, dataFrame["per capita income (us$)"])

predictedYear = input("Enter a Predicted Year: ")
predictedYearDataFrame = pd.DataFrame({"year":[predictedYear]})
predictionModel = model.predict(predictedYearDataFrame)

print("================================================")
print(f"Canada per Capita Income in {predictedYear} is {predictionModel[0]:.2f}")
print("================================================")



print("ML Model Run...")