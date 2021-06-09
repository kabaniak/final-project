import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd



# methods
def plot_by_year(targs, pred, err, train, deg):
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


    return final_yr, final_acc


head = np.arange(91)
load = pd.read_csv('YearPredictionMSD.txt', delimiter = ",", header = None, names = head)


data = load.to_numpy()

# all data, tr test split
trData = data[0:463715, 1:]
teData = data[463715:, 1:]

trTarg = data[0:463715, 0]
teTarg = data[463715:, 0]

avg_trData = trData[:, 0:12]
avg_teData = teData[:, 0:12]

# train model on timbre average
model_all = LinearRegression().fit(avg_trData, trTarg)

pred_all = model_all.predict(avg_teData)

# completely acc
fig1 = plt.figure()
fir, fir_acc = plot_by_year(teTarg, pred_all, 0, trTarg, 1)
plt.title('Accuracy vs. Degree (Average Timbre)')
plt.xlabel('Year')
plt.ylabel('Accuracy')


yrs = list()
accs = list()
for i in range(2, 5):
    print("Doing degree ", i)
    poly = PolynomialFeatures(degree = i)
    tr = poly.fit_transform(avg_trData)
    te = poly.fit_transform(avg_teData)

    model = LinearRegression().fit(tr, trTarg)
    pred = model.predict(te)

    yr, acc = plot_by_year(teTarg, pred, 0, trTarg, i)
    yrs.append(yr)
    accs.append(acc)
    

plt.plot(fir, fir_acc, label = "1st Degree")
plt.plot(yrs[0], accs[0], label = "2nd Degree")
plt.plot(yrs[1], accs[1], label = "3rd Degree")
plt.plot(yrs[2], accs[2], label = "4th Degree")
plt.legend(loc ='best')

plt.show()
    

plt.show()

