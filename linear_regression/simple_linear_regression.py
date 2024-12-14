"""
Singel Variable Linear Regression

Or

Simple Linear Regression
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model



dataFrame = pd.read_csv("/home/ubuntu/Documents/machine-learning/home_prices.csv")

dataFrame.columns = dataFrame.columns.str.strip().str.lower()

x = dataFrame.area
y = dataFrame.price

m, b = np.polyfit(x, y, 1)

plt.figure(figsize=(8, 6))

plt.scatter(x, y, color="red", marker="o", label="Data Points")
plt.plot(x, x * m + b, color="blue", label="Regression Line")
plt.plot(x, y, color="magenta", label="Prices up-downs")

plt.xlabel("Area(Square Foot)")
plt.ylabel("Price(USD$)")
plt.title("Home Prices vs. Area")

plt.legend()
plt.savefig("home_prices.jpg")


newDataFrame = dataFrame.drop("price", axis="columns")

model = linear_model.LinearRegression()
model.fit(newDataFrame, dataFrame.price)

predictedArea = int(input("Enter an Area to predicted: "))

areaToPredict = pd.DataFrame({"area": [predictedArea]})
predictionModel = model.predict(areaToPredict)
print("================================================")
print(f"{predictedArea} (sq. foot) Area Home Price is {predictionModel[0]:.2f}")
print("================================================")


# coef = model.coef_[0]
# intercept = model.intercept_
# print(f"y = x * m + b | y = {predictedArea} * {coef} + {intercept} | y = {predictedArea * coef + intercept:.2f}")




# ================== Area Data Frame to predecticated ==================


areaDataFrame = pd.read_csv("/home/ubuntu/Documents/machine-learning/areas.csv")

areaDataFrame.columns = areaDataFrame.columns.str.strip().str.lower()

predictionPrices = model.predict(areaDataFrame)

areaDataFrame["price"] = np.round(predictionPrices, 2)

areaDataFrame.to_csv("predicted_area_df_home_price.csv")



print("ML Model Run...")