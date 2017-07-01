
### multilayer perceptron algorithm
the multilayer perceptron take iterations, initial learning_rate, hidden_layer_dim (the size of each hidden layer e.g [300,300]), activation_functions and use backpropagation as the optimization algorithm. It will decay learning rate by half and restore the last iteration weights and bias if the learning rate leads to larger loss function. 

### example of one layer perceptron: 

my_MLP = MLP_Classifier(X=train_data, iterations=800, learning_rate=0.005, hidden_layer_dim=[100], activation_functions=['ReLU'])
my_MLP.fit(train_data[:,0:d], train_data[:,d].astype(int))
my_MLP.predict(test_data[:,0:d], test_data[:,d].astype(int))

### example of two layers perceptron: 
my_MLP = MLP_Classifier(X=train_data, iterations=800, learning_rate=0.001, hidden_layer_dim=[300,300], activation_functions=['ReLU','ReLU'])
my_MLP.fit(train_data[:,0:d], train_data[:,d].astype(int))
my_MLP.predict(test_data[:,0:d], test_data[:,d].astype(int))

### result
both will get around 93% accuracy on iris dataset.
