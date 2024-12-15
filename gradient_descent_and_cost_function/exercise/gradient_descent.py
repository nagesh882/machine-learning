import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def gradient_descent(x, y):
    
    m_curr = 0
    b_curr = 0
    n = len(x)
    iteration = 10000
    rate = 0.01
    cost_previous = float('inf')
    plt.scatter(x, y, color="red", marker="o", linewidths=5)
    for i in range(iteration):
        yp = x * m_curr + b_curr
        cost = (1 / n) * sum((y - yp) ** 2)
        md = -(2/n) * sum(x * (y - yp))
        bd = -(2/n) * sum(y - yp)
        m_curr = m_curr - rate * md
        b_curr = b_curr - rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-9):
            break
        cost_previous = cost

        if i % 500 == 0:
            plt.plot(x, yp, color='green', alpha=0.5)
        print(f"Iteration {i}: m={m_curr:.5f}, b={b_curr:.5f}, cost={cost:.5f}")

    plt.plot(x, yp, color='blue', label='Best Fit Line')
    plt.xlabel("Math Scores")
    plt.ylabel("CS Scores")
    plt.title("Gradient Descent Visualization")
    plt.legend()
    plt.savefig("test_score_gradient_descent.jpg")


dataFrame = pd.read_csv("/home/ubuntu/Documents/machine-learning/test_scores.csv")

dataFrame.columns = dataFrame.columns.str.strip().str.lower()

x = np.array(dataFrame["math"])
y = np.array(dataFrame["cs"])

gradient_descent(x, y)



print("ML Model Run...")