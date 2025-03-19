import random
import numpy as np
import dataset as ds


# y_true = wx + b
w = random.uniform(-1, 1)
b = random.uniform(-1, 1)
α = 0.01

train_y_pred = np.array(list((w*carat + b) for carat in ds.X_train)).flatten()

sqr_error = (train_y_pred - ds.y_train) ** 2

dMSE_dw = 2 * np.mean(ds.X_train * (train_y_pred - ds.y_train) )
w -= α*dMSE_dw
dMSE_db = 2  * np.mean(train_y_pred - ds.y_train)
b -= α * dMSE_db

#test
y_pred = np.array(list( (w*carat +b) for carat in ds.X_test)).flatten()

sqr_error = (y_pred - ds.y_test) ** 2

MSE = np.mean(sqr_error)
variance = np.var(ds.y_test)
r2 = 1 - (MSE / variance)


count = 0
while r2 < 0.8:
    dMSE_dw = 2 * np.mean(ds.X_test * (y_pred - ds.y_test))
    w -= α * dMSE_dw
    dMSE_db = 2  * np.mean((y_pred - ds.y_test))
    b -= α * dMSE_db


    y_pred = np.array(list((w*carat + b) for carat in ds.X_test)).flatten()

    sqr_error = (y_pred - ds.y_test) ** 2

    MSE = np.mean(sqr_error)
    variance = np.var(ds.y_test)

    r2 = 1 - (MSE / variance)

    count += 1
    if count == 1500:
        break


