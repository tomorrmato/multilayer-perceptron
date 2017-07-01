import numpy as np
np.random.seed(0)

train_data=np.genfromtxt("data/iris_training.csv", delimiter=',', skip_header=1, dtype=np.float32)
test_data=np.genfromtxt("data/iris_test.csv", delimiter=',', skip_header=1, dtype=np.float32)

d = 4 # data (x_train and x_test) dimension 
k = 3 # number of classes
lamda = 0.005 # regularization constant

batch_size = 20
start_index, end_index = 0, batch_size 
batch_end = False

def get_batch(self, data): # load data batch by batch
	global start_index, end_index, batch_end

	if end_index <= training_examples:
		start_index += batch_size
		end_index += batch_size
		batch_end = False
		return data[(start_index-batch_size):(end_index-batch_size)]
	else:
		start_index, end_index = 0, batch_size
		batch_end = True

# activation function
def activate(x, func):
	if func=='sigmoid':
		return 1/(1+np.exp(-x))
	elif func=='ReLU':
		return np.maximum(0, x)
	elif func=='Tanh':
		return np.tanh(x)
	else:
		return x

# derivative of activation function
def d_activation(x, func):
	if func=='sigmoid':
		return x*(1-x)
	elif func=='ReLU':
		x[x<=0]=0
		return x
	elif func=='Tanh':
		return 1-np.tanh(x)**2
	else:
		return x

# softmax of last layer output 
def softmax(x):
	xx = (x.T-np.amax(x,axis=1)).T
	exp_x = np.exp(xx)
	return exp_x / np.sum(exp_x, axis=1, keepdims=True)
	

class MLP_Classifier(object):

	def __init__(self, X, iterations, learning_rate, hidden_layer_dim, activation_functions):		
		self.X = X
		self.iterations = iterations
		self.learning_rate = learning_rate
		self.hidden_layer_dim = hidden_layer_dim
		self.activation_functions = activation_functions
		self.restore_state = False

		# initialize weights, bias, hidden layer and layer output
		weight_mat = [] # w[i] is the weight matrix w_i
		bias_vec = [] # b[i] is the bias vector b_i
		hidden_layer = []  
		layer_output = [] # output of hidden layer
		
		self.num_layers = len(hidden_layer_dim)
		for i in range(self.num_layers): 
			curr_activation_func = self.activation_functions[i]

			if i==0:
				current_weight = np.random.normal(0, 0.1, size=(d,hidden_layer_dim[i]))
				current_bias = np.random.normal(0, 0.1, hidden_layer_dim[i])
				weight_mat.append(current_weight)
				bias_vec.append(current_bias)
				curr_hidden_layer = np.dot(self.X[:,:d], current_weight)+current_bias
				curr_layer_output = activate(curr_hidden_layer, curr_activation_func)
				hidden_layer.append(curr_hidden_layer)
				layer_output.append(curr_layer_output)

			if i > 0: 
				current_weight = np.random.normal(0, 0.1, size=(hidden_layer_dim[i-1],hidden_layer_dim[i]))
				current_bias = np.random.normal(0, 0.1, hidden_layer_dim[i])
				weight_mat.append(current_weight)
				bias_vec.append(current_bias)
				curr_hidden_layer = np.dot(curr_layer_output, current_weight)+current_bias
				curr_layer_output = activate(curr_hidden_layer, curr_activation_func)
				hidden_layer.append(curr_hidden_layer)
				layer_output.append(curr_layer_output)

			if i==(len(hidden_layer_dim)-1): 
				current_weight = np.random.normal(0, 0.1, size=(hidden_layer_dim[i],k))
				current_bias = np.random.normal(0, 0.1, k)
				weight_mat.append(current_weight)
				bias_vec.append(current_bias)
				self.final_layer_output = np.dot(curr_layer_output, current_weight)+current_bias
				self.est_prob = softmax(self.final_layer_output)
						
		self.weight_mat = self.weight_mat_backup = weight_mat # num_layers+1
		self.bias_vec = self.bias_vec_backup = bias_vec # num_layers+1
		self.hidden_layer = hidden_layer # num_layers
		self.layer_output = layer_output # num_layers

	# forwardfeed process
	def forwardfeed(self, x_train, y_train): 
		num_examples = len(x_train)
		for i in range(self.num_layers):
			curr_activation_func = self.activation_functions[i]
			if i == 0:
				self.hidden_layer[i] = np.dot(x_train, self.weight_mat[i])+self.bias_vec[i]
				self.layer_output[i] = activate(self.hidden_layer[i], curr_activation_func)
			if i > 0:
				self.hidden_layer[i] = np.dot(self.layer_output[i-1], self.weight_mat[i])+self.bias_vec[i]
				self.layer_output[i] = activate(self.hidden_layer[i], curr_activation_func)
			if i == (self.num_layers-1):
				self.final_layer_output = np.dot(self.layer_output[i], self.weight_mat[i+1])+self.bias_vec[i+1]
				self.est_prob = softmax(self.final_layer_output)
		# cross entropy
		corect_logprobs = -np.log(np.maximum(self.est_prob[range(num_examples),y_train], 0.00001))
		# data loss
		self.data_loss = np.sum(corect_logprobs)/num_examples
		# regularization loss
		reg_loss = 0
		for i in range(len(self.weight_mat)):
			reg_loss += np.sum(self.weight_mat[i]*self.weight_mat[i])
		reg_loss *= lamda
		self.reg_loss = reg_loss
		# total loss
		self.loss = self.data_loss+self.reg_loss

		return np.argmax(self.est_prob, axis=1)

	# backpropagation process 
	def backpropagate(self, x_train, y_train): 
		num_examples = len(x_train)
		self.d_softmax = self.est_prob
		# derivative of softmax
		self.d_softmax[range(num_examples), y_train] -= 1 
		self.d_softmax /= num_examples # gradient from softmax 

		# gradient at layer i
		delta = self.num_layers*[[]] #gradient at hidden layer i

		# from last layer to first layer
		for i in reversed(range(self.num_layers)):
			curr_activation_func = self.activation_functions[i]

			if i == (self.num_layers-1): 
				d_weight_last = np.dot(self.layer_output[i].T, self.d_softmax)
				d_bias_last = np.sum(self.d_softmax, axis=0)
				self.weight_mat[i+1] -= self.learning_rate*(d_weight_last+lamda*self.weight_mat[i+1])
				self.bias_vec[i+1] -= self.learning_rate*d_bias_last
				delta[i] = d_activation(self.hidden_layer[i], curr_activation_func)*np.dot(self.d_softmax, self.weight_mat[i+1].T)
			if i > 0:
				d_weight_i = np.dot(self.layer_output[i-1].T, delta[i])
				d_bias_i = np.sum(delta[i], axis=0)
				delta[i-1] = np.dot(delta[i], self.weight_mat[i].T)*d_activation(self.hidden_layer[i-1], curr_activation_func)
				self.weight_mat[i] -= self.learning_rate*(d_weight_i+lamda*self.weight_mat[i])
				self.bias_vec[i] -= self.learning_rate*(d_bias_i)
			if i == 0:
				d_weight_init = np.dot(x_train.T, delta[i])
				d_bias_init = np.sum(delta[i], axis=0)
				self.weight_mat[i] -= self.learning_rate*(d_weight_init+lamda*self.weight_mat[i])
				self.bias_vec[i] -= self.learning_rate*(d_bias_init)

	# restore to last step if the learning rate lead to incresed loss 
	def restore(self):
		if self.restore_state:
			self.weight_mat = self.weight_mat_backup
			self.bias_vec = self.bias_vec_backup
		else:
			self.weight_mat_backup = self.weight_mat
			self.bias_vec_backup = self.bias_vec

	# the learning process
	def fit(self, x_train, y_train):

		for i in range(self.iterations):	
			if i == 0:
				self.backpropagate(x_train, y_train)
				self.forwardfeed(x_train, y_train)
				self.restore_state = False
				self.restore()
				last_loss = self.loss
			else:
				self.backpropagate(x_train, y_train)
				self.forwardfeed(x_train, y_train)
				if self.loss >= last_loss:
					print 'last and current loss: %s %s' % (last_loss, self.loss)
					print '-------decaying learning rate------'
					# decay learning rate by half if it is too large
					self.learning_rate *= 0.5
					print self.learning_rate
					# restore weight and bias to last iteration and use new learning rate
					if self.learning_rate < 1e-10:
						print 'learning rate is too small to stop, try other learning rate'
						break
					self.restore_state = True
					self.restore()
				else:
					self.restore_state = False
					self.restore()
					last_loss = self.loss
		
				if i % 50 == 0:
					print 'i = %s, loss = %s, data loss = %s, reg loss = %s, accuracy = %s ' %(i, self.loss, self.data_loss, self.reg_loss, np.mean(np.argmax(self.est_prob, axis=1)==y_train))
			

	def predict(self, x_test, y_test):
		self.forwardfeed(x_test, y_test)
		print 'accuracy in testing data = %s' % np.mean(np.argmax(self.est_prob, axis=1)==y_test)

if __name__ == '__main__':
	# single layer perceptron
	my_MLP = MLP_Classifier(X=train_data, iterations=800, learning_rate=0.005, hidden_layer_dim=[100], activation_functions=['ReLU'])
	my_MLP.fit(train_data[:,0:d], train_data[:,d].astype(int))
	my_MLP.predict(test_data[:,0:d], test_data[:,d].astype(int))

	# multilayer perceptron
	#my_MLP = MLP_Classifier(X=train_data, iterations=800, learning_rate=0.001, hidden_layer_dim=[300,300], activation_functions=['ReLU','ReLU'])
	#my_MLP.fit(train_data[:,0:d], train_data[:,d].astype(int))
	#my_MLP.predict(test_data[:,0:d], test_data[:,d].astype(int))



