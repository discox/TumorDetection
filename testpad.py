
def mynewfunction(var1, var2):

    # load libraries
    import numpy
    from numpy import genfromtxt
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier



    # open file and load data from https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    y_temp = genfromtxt('wdbc.data', delimiter=',', usecols=1, dtype='|S1')
    X = genfromtxt('wdbc.data', delimiter=',', usecols=range(2,32) )


    # extract the data
    y = numpy.zeros((y_temp.shape))

    for i in range(0,y_temp.size-1):
            if y_temp[i] == b'M':
                y[i] = 1

    return var1 + var2


print(mynewfunction(5,4))