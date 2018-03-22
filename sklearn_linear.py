import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

data = [
    [0.7,0.15],
    [0.48,0.19],
    [0.6,0.2],
    [0.35,0.25],
    [0.25,0.25],
    [0.15,0.34],
    [0.51,0.51],
    [0.42,0.42],
    [0.3,0.34],
    [0.23,0.45],
    [0.35,0.47],
    [0.58,0.48],
    [0.47,0.6],
    [0.59,0.59],
    [0.66,0.6],
    [0.61,0.7],
    [0.55,0.75],
    [0.46,0.71],
    [0.31,0.78],
    [0.15,0.81],
    [0.12,0.8]
]
y = list()
x = list()
for elt in data:
    y.append(elt[0])
    x.append(elt[1])


regr = linear_model.LinearRegression()
regr.fit(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))
inter = float(regr.intercept_)
slope = float(regr.coef_)
print regr.score(np.array(x).reshape(-1,1),np.array(y).reshape(-1,1))
print inter,slope

ablin = [slope*i + inter for i in range(10)]
print ablin
plt.axis([0, 0.9, 0, 0.9])
plt.scatter(x,y)
plt.plot(ablin)
plt.show()
