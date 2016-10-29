
from sklearn.neural_network import MLPClassifier

# data input and output for training
X = [[0., 0.], [1., 1.]]
y = [0, 1]


#defining the model
clf = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

# training the model
para = clf.fit(X, y)
print("the paramaters are",para)


#predict a new output
pred = clf.predict([[2., 2.], [-1., -2.]])
print(pred)





######


print("program finished")

