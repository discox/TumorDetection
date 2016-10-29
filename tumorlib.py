
# packaged the tumor program as a function of the number of neurons in layers 1 and 2 and as the random initiators
# for the backpropagation algorithms.

def tumorfunc(layer1,layer2,ran_split, ran_init):

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


    ############################## split train and test datasets ##############################

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=ran_split)


    ############################## data normalisation ##############################





    ############################## prep the model ##############################

    clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(layer1,layer2), random_state=ran_init)
    #clf = MLPClassifier(algorithm='adam', alpha=1e-10, hidden_layer_sizes=(100,100,100), random_state=1989, learning_rate='adaptive')



    # train the data
    para = clf.fit(X_train, y_train)



    # test the model

    output = numpy.zeros((y_test.shape))


    for trav in range(X_test.shape[0]):
        output[trav] = clf.predict(X_test[trav].reshape(1,-1))


    compare = (y_test==output)

    return numpy.sum(compare)/compare.size

