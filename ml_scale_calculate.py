import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression #using linearregression
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets #using RANSAC
import pickle #保存模型



c = pd.read_csv('markers_distance.csv', usecols=['x','y']).values
d = pd.read_csv('markers_distance.csv', usecols=['ids']).values
x = pd.read_csv('markers_distance.csv', usecols=['x']).values
y = pd.read_csv('markers_distance.csv', usecols=['y']).values
d_y = pd.read_csv('markers_distance.csv', usecols=['distance_x']).values
d_x = pd.read_csv('markers_distance.csv', usecols=['distance_y']).values

#c_test = pd.read_csv('markers_data.csv', usecols=['xt','yt']).values
#xt = pd.read_csv('markers_data.csv', usecols=['xt']).values
#yt = pd.read_csv('markers_data.csv', usecols=['yt']).values


###linear regression
# train model
lm = LinearRegression()
lm.fit(c, d)
# 计算R平方
lm.score(c, d)
print(lm.coef_)
# 印出截距
print(lm.intercept_ )
# 计算y_hat1
#y_hat = lm.predict(c_test)

lm.fit(c, d)
# 存储模型
with open('./linear1.pkl', 'wb') as f:
    pickle.dump(lm, f)
    print("save success！")

# 打印出图
plt.scatter(x, d)
#plt.plot(xt, y_hat, color='r')
###linear regression

###ransac
# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(x, d_x)
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(x, d_x)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
# Predict data of estimated models
line_X = np.arange(x.min(), x.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)
# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print(lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(
    x[inlier_mask], d_x[inlier_mask], color="yellowgreen", marker=".", label="Inliers-x"
)
plt.scatter(
    x[outlier_mask], d_x[outlier_mask], color="gold", marker=".", label="Outliers-x"
)
plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor-x")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor-x",
)
lr.fit(x, d_x)
# 存储模型
with open('./linear2_x.pkl', 'wb') as f:
    pickle.dump(ransac, f)
    print("save success！")
###ransac for x

###ransac
# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(y, d_y)
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(y, d_y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
# Predict data of estimated models
line_X = np.arange(y.min(), y.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print(lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(
    y[inlier_mask], d_y[inlier_mask], color="green", marker=".", label="Inliers-y"
)
plt.scatter(
    y[outlier_mask], d_y[outlier_mask], color="yellow", marker=".", label="Outliers-y"
)
plt.plot(line_X, line_y, color="red", linewidth=lw, label="Linear regressor-y")
plt.plot(
    line_X,
    line_y_ransac,
    color="black",
    linewidth=lw,
    label="RANSAC regressor-y",
)

plt.legend(loc='lower left')

lr.fit(y, d_y)
# 存储模型
with open('./linear2_y.pkl', 'wb') as f:
    pickle.dump(ransac, f)
    print("save success！")


###ransac for y


#plot 图
plt.show()
