import numpy as np
import matplotlib.pyplot as plt
import xlrd


src = 'DATA.xls'

wkb = xlrd.open_workbook(src)

targetsheet = wkb.sheet_by_index(1)

datasheet = wkb.sheet_by_index(0)

learn_rate = 0.01




def uncode(target):
    for row in range(len(target)):
        minimum = abs(target[row] - 0.333)
        kod = 1
        if (minimum > abs(target[row] - 0.666)):
            minimum = abs(target[row] - 0.666)
            kod = 2
        if (minimum > abs(target[row] - 1)):
            minimum = abs(target[row] - 1)
            kod = 3

        target[row] = kod

    return target


def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))


X = np.zeros(shape=(datasheet.nrows, datasheet.ncols))
Y = np.zeros(shape=(targetsheet.nrows, targetsheet.ncols))



for row in range(targetsheet.nrows):
    for col in range(targetsheet.ncols):
        Y[row,col] = targetsheet.cell_value(row,col)

for row in range(datasheet.nrows):
    for col in range(datasheet.ncols):
        X[row,col] = datasheet.cell_value(row,col)

np.random.seed(1)

weights0 = 2*np.random.random((len(X[0,:]),len(X[:,0]))) - 1
weights1 = 2*np.random.random((len(Y[:,0]),len(Y[0,:]))) - 1


#wykres
xdata = []
ydata = []

plt.show()
axes = plt.gca()

Error = 1
j = 0



while Error > 0.01:
    layer0 = X
    layer1 = nonlin(np.dot(layer0,weights0))
    layer2 = nonlin(np.dot(layer1,weights1))

    layer2_error = Y - layer2

    Error = np.mean(np.abs(layer2_error))
    if (j%10000) == 0:
        print("Error:"+ str(Error))

    #Back propagation of errors using the chain rule
    layer2_delta = layer2_error * nonlin(layer2, deriv=True)
    layer1_error = layer2_delta.dot(weights1.T)
    layer1_delta = layer1_error * nonlin(layer1, deriv=True)

    #Using the deltas, we can use them to update the weights to reduce the error rate with every iteration
    #This algoritm is called gradient descent
    weights1 += layer1.T.dot(layer2_delta) * learn_rate # 0.01 is our learning rate
    weights0 += layer0.T.dot(layer1_delta) * learn_rate

    xdata.append(j)
    ydata.append(np.mean(np.abs(layer2_error)))
    j = j+1



plt.semilogx(xdata, ydata, label='linear')
plt.show()





print "Output after training"
print uncode(layer2)
print j