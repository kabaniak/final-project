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
        final_yr.append(yrs[i])

    plt.title('Accuracy by Year')
    plt.xlabel('Year')
    plt.ylabel('Accuracy')
    plt.plot(final_yr, final_acc)
    #plt.show()

def plot_correct_decade(targs, pred):
    yrs = np.arange(10)
    acc_by_year = np.zeros(len(yrs))
    tot_by_year = np.zeros(len(yrs))
    for i in range(len(targs)):
        act = get_decade(targs[i])
        pred_dec = get_decade(pred[i])
        if act == pred_dec:
            acc_by_year[act] += 1
        tot_by_year[act] += 1

    final_acc = list()
    final_yr = list()
    for i in range(len(acc_by_year)):
        if tot_by_year[i] == 0: continue
        final_acc.append(acc_by_year[i] / tot_by_year[i])
        final_yr.append(yrs[i])

    locs = [1, 3, 5,  7, 9]
    label = ["1930s", "1950s", "1970s", "1990s", "2010s"]
    plt.title('Accuracy by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Accuracy')
    plt.xticks(ticks = locs, labels = label)
    plt.plot(final_yr, final_acc)
    #plt.show()

def targ_to_dec(testT, trainT):
    final_test = np.zeros(len(testT))
    for i in range(len(testT)):
        final_test[i] = int(testT[i] / 10) * 10
        
    final_train = np.zeros(len(trainT))
    for i in range(len(trainT)):
        final_train[i] = int(trainT[i] / 10) * 10

    return final_train, final_test

def heat_map_yrs(targs, preds):
    yrs = np.arange(10)
    heat = np.zeros((len(yrs), len(yrs)))
    tot = np.zeros(len(yrs))

    for i in range(len(targs)):
        yrRnd = round(preds[i])
        pred_yr_ind = get_decade(preds[i])
        act_yr_ind = get_decade(targs[i])
        if pred_yr_ind >= len(tot):
            tot[int(act_yr_ind)] += 1
            continue
        if pred_yr_ind < 0:
            tot[int(act_yr_ind)] += 1
            continue
        heat[int(pred_yr_ind)][int(act_yr_ind)] += 1
        tot[int(act_yr_ind)] += 1

    for i in range(len(heat)):
        if tot[i] == 0: continue
        for j in range(len(heat)):
            heat[j][i] = heat[j][i] / tot[i]
    heat = np.flip(heat, axis = 0)

    locs = [1, 3, 5,  7, 9]
    locsy = [8, 6, 4, 2, 0]
    label = ["1930s", "1950s", "1970s", "1990s", "2010s"]
    plt.title("Heatmap of Actual Decade vs. Predicted Decade")
    plt.xlabel("Actual Decade")
    plt.ylabel("Predicted Decade")
    plt.yticks(ticks = locsy, labels = label)
    plt.xticks(ticks = locs, labels = label)
    plt.imshow(heat)
    plt.colorbar()

def get_decade(val):
    dec = int( val / 10 ) * 10
    return int ((dec - 1920) / 10)


head = np.arange(91)
load = pd.read_csv('YearPredictionMSD.txt', delimiter = ",", header = None, names = head)


data = load.to_numpy()

# all data, tr test split
trData = data[0:463715, 1:]
teData = data[463715:, 1:]

trTarg = data[0:463715, 0]
teTarg = data[463715:, 0]

# convert targets to decades
tr_decTarg, te_decTarg = targ_to_dec(teTarg, trTarg)

# train models
model_all = LinearRegression().fit(trData, trTarg)
model_bydec = LinearRegression().fit(trData, tr_decTarg)

pred_all = model_all.predict(teData)
pred_bydec = model_bydec.predict(teData)

# graph
fig1 = plt.figure()
plot_correct_decade(te_decTarg, pred_bydec)
plt.show()

fig2 = plt.figure()
heat_map_yrs(te_decTarg, pred_bydec)
plt.show()

