
the multilayer perceptron take iterations, initial learning_rate, hidden_layer_dim (the size of each hidden layer e.g [300,300]), activation_functions and use backpropagation as the optimization algorithm. It will decay learning rate by half and restore the last iteration weights and bias if the learning rate leads to larger loss function. 

example of one layer perceptron: 

# single layer perceptron
my_MLP = MLP_Classifier(X=train_data, iterations=800, learning_rate=0.005, hidden_layer_dim=[100], activation_functions=['ReLU'])
my_MLP.fit(train_data[:,0:d], train_data[:,d].astype(int))
my_MLP.predict(test_data[:,0:d], test_data[:,d].astype(int))

output:
i = 50, loss = 0.938015485294, data loss = 0.902610603824, reg loss = 0.0354048814699, accuracy = 0.7 
i = 100, loss = 0.838348560355, data loss = 0.801360478865, reg loss = 0.0369880814907, accuracy = 0.7 
i = 150, loss = 0.752658988644, data loss = 0.713377175018, reg loss = 0.0392818136257, accuracy = 0.7 
i = 200, loss = 0.68078131564, data loss = 0.638575567603, reg loss = 0.0422057480371, accuracy = 0.7 
i = 250, loss = 0.623071438807, data loss = 0.577430524834, reg loss = 0.0456409139733, accuracy = 0.7 
i = 300, loss = 0.578099802529, data loss = 0.528641112385, reg loss = 0.0494586901442, accuracy = 0.725 
i = 350, loss = 0.543344421807, data loss = 0.489789255793, reg loss = 0.0535551660145, accuracy = 0.75 
i = 400, loss = 0.516169173993, data loss = 0.458303741406, reg loss = 0.0578654325874, accuracy = 0.825 
i = 450, loss = 0.494424221153, data loss = 0.432063904145, reg loss = 0.062360317008, accuracy = 0.858333333333 
i = 500, loss = 0.476526740153, data loss = 0.409490574239, reg loss = 0.0670361659145, accuracy = 0.858333333333 
i = 550, loss = 0.46140738503, data loss = 0.38950255843, reg loss = 0.0719048266008, accuracy = 0.891666666667 
i = 600, loss = 0.448346725543, data loss = 0.371360502865, reg loss = 0.0769862226783, accuracy = 0.925 
i = 650, loss = 0.436875563937, data loss = 0.354572527646, reg loss = 0.0823030362909, accuracy = 0.933333333333 
i = 700, loss = 0.426700433592, data loss = 0.338823241606, reg loss = 0.0878771919862, accuracy = 0.95 
i = 750, loss = 0.417640969259, data loss = 0.323913417935, reg loss = 0.0937275513234, accuracy = 0.958333333333 
accuracy in testing data = 0.933333333333

