import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd



# methods
def plot_by_year(targs, pred, err, train, trpreds):
    yrs = np.arange(1922, 2012)
    acc_by_year = np.zeros(len(yrs))
    tot_by_year = np.zeros(len(yrs))

    test_samps = np.zeros(len(yrs))
    
    for i in range(len(targs)):
        yrRnd = round(pred[i])
        yr_ind = targs[i] - 1922
        lowerr = yrRnd - err
        higherr = yrRnd + err
        if higherr >= targs[i] and lowerr <= targs[i]:
            acc_by_year[int(yr_ind)] += 1
        tot_by_year[int(yr_ind)] += 1

    final_acc = list()
    final_yr = list()
    for i in range(len(acc_by_year)):
        if tot_by_year[i] == 0: continue
        final_acc.append(acc_by_year[i] / tot_by_year[i])
        final_yr.append(yrs[i])
        
    plt.title('Accuracy of Predicting Train vs. Test')
    plt.xlabel('Year')
    plt.ylabel('Accuracy')
    plt.plot(final_yr, final_acc, label = "Testing Accuracy")

    acc_tr = np.zeros(len(yrs))
    tot_tr = np.zeros(len(yrs))

    for i in range(len(train)):
        yrRnd = round(trpreds[i])
        yr_ind = train[i] - 1922
        lowerr = yrRnd - err
        higherr = yrRnd + err
        if higherr >= train[i] and lowerr <= train[i]:
            acc_tr[int(yr_ind)] += 1
        tot_tr[int(yr_ind)] += 1

    final_acc = list()
    final_yr = list()
    for i in range(len(acc_tr)):
        if tot_tr[i] == 0: continue
        final_acc.append(acc_tr[i] / tot_tr[i])
        final_yr.append(yrs[i])
    
    plt.plot(final_yr, final_acc, label = "Training Accuracy")
    plt.legend(loc ='best')


head = np.arange(91)
load = pd.read_csv('YearPredictionMSD.txt', delimiter = ",", header = None, names = head)


data = load.to_numpy()

# all data, tr test split
trData = data[0:463715, 1:]
teData = data[463715:, 1:]

trTarg = data[0:463715, 0]
teTarg = data[463715:, 0]

# train models
model_all = LinearRegression().fit(trData, trTarg)

pred_all = model_all.predict(teData)
pred_tr = model_all.predict(trData)

# completely acc
fig1 = plt.figure()
plot_by_year(teTarg, pred_all, 0, trTarg, pred_tr)

plt.show()

