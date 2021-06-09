import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd



# methods
def plot_by_year(targs, pred, err, train):
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
        test_samps[int(yr_ind)] += 1

    final_acc = list()
    final_yr = list()
    for i in range(len(acc_by_year)):
        if tot_by_year[i] == 0: continue
        final_acc.append(acc_by_year[i] / tot_by_year[i])
        final_yr.append(yrs[i])

    train_samps = np.zeros(len(yrs))

    for i in range(len(train)):
        yr_ind = train[i] - 1922
        train_samps[int(yr_ind)] += 1


    for i in range(len(yrs)):
        test_samps[i] = test_samps[i] / len(targs)
        train_samps[i] = train_samps[i] / len(train)
        
    print(np.sum(train_samps))
    plt.title('Accuracy vs. Proportion of Samples')
    plt.xlabel('Year')
    plt.ylabel('Accuracy')
    plt.plot(final_yr, final_acc, label = "% Accurately Classified")
    plt.plot(yrs, train_samps, label = "% of Training Sample")
    plt.plot(yrs, test_samps, label = "% of Testing Sample")
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

# completely acc
fig1 = plt.figure()
plot_by_year(teTarg, pred_all, 0, trTarg)

plt.show()

