import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd



# methods
def data_yr_range(trainD, trainT, testD, testT, start, end):
    final_trainD = list()
    final_trainT = list()
    for i in range(len(trainD)):
        if start <= trainT[i] and end >= trainT[i]:
            final_trainD.append(trainD[i])
            final_trainT.append(trainT[i])

    final_testD = list()
    final_testT = list()
    for i in range(len(testD)):
        if start <= testT[i] and end >= testT[i]:
            final_testD.append(testD[i])
            final_testT.append(testT[i])

    return final_trainD, final_trainT, final_testD, final_testT

def plot_by_year(targs, pred, err):
    yrs = np.arange(1922, 2012)
    acc_by_year = np.zeros(len(yrs))
    tot_by_year = np.zeros(len(yrs))
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
        final_yr.append(int(yrs[i]))

    plt.xlabel('Year')
    plt.ylabel('Accuracy')

    locs = list()
    for i in range(len(final_yr)):
        if i % 2 == 0:
            locs.append(final_yr[i])
    plt.xticks(ticks = locs)
    plt.plot(final_yr, final_acc)
    #plt.show()

def heat_map_yrs(preds, targs, start, end):
    yrs = np.arange(start, end + 1)
    heat = np.zeros((len(yrs), len(yrs)))
    tot = np.zeros(len(yrs))

    for i in range(len(targs)):
        yrRnd = round(preds[i])
        pred_yr_ind = end - yrRnd
        act_yr_ind = targs[i] - start 
        if pred_yr_ind >= len(tot):
            pred_yr_ind = len(tot) - 1
        if pred_yr_ind < 0:
            pred_yr_ind = 0
        heat[int(pred_yr_ind)][int(act_yr_ind)] += 1
        tot[int(act_yr_ind)] += 1

    for i in range(len(heat)):
        continue
        if tot[i] == 0: continue
        for j in range(len(heat)):
            heat[j][i] = heat[j][i] / tot[i]

    locs = list()
    locsy = list()
    label = list()
    for i in range(len(yrs)):
        if i % 2 == 0:
            locs.append(i)
            locsy.append(i + 1)
            label.append(yrs[i])
    plt.xlabel("Actual Year")
    plt.ylabel("Predicted Year")
    plt.yticks(ticks = np.flipud(locsy), labels = label)
    plt.xticks(ticks = locs, labels = label)
    plt.imshow(heat)
    plt.colorbar()


head = np.arange(91)
load = pd.read_csv('YearPredictionMSD.txt', delimiter = ",", header = None, names = head)


data = load.to_numpy()

# all data, tr test split
trData = data[0:463715, 1:]
teData = data[463715:, 1:]

trTarg = data[0:463715, 0]
teTarg = data[463715:, 0]


for i in range(10):
    start = 1920 + i*10
    end = start + 9

    tr_rangeD, tr_rangeT, te_rangeD, te_rangeT = data_yr_range(trData, trTarg, teData, teTarg, start, end)

    model_range = LinearRegression().fit(tr_rangeD, tr_rangeT)
    preds_range = model_range.predict(te_rangeD)

    """figure1 = plt.figure()
    title = 'Accuracy by Year - ' + str(start) + 's, +/- a Year'
    plt.title(title)
    plot_by_year(te_rangeT, preds_range, 1)
    plt.show()"""

    figure2 = plt.figure()
    title = 'Heatmap of Prediction vs Actual - ' + str(start) + 's'
    plt.title(title)
    heat_map_yrs(preds_range, te_rangeT, start, end)
    plt.show()
