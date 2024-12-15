"""
Gradient Descent and Cost Function
"""


import numpy as np
import matplotlib.pyplot as plt



def gradient_descent(x, y):

    m_curr = b_curr = 0
    iteration = 10000
    n = len(x)
    learning_rate = 0.01
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="red", marker="o", linewidths=5)
    for i in range(iteration):
        yp = x * m_curr + b_curr
        plt.plot(x,yp,color='green')
        cost = (1/n) * sum([val**2 for val in (y - yp)])
        md = -(2/n) * sum(x * (y - yp))
        bd = -(2/n) * sum(y - yp)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Gradient Descent Visualization")
    plt.savefig("gradient_descent.jpg")


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])


gradient_descent(x, y)



print("ML Model Run...")