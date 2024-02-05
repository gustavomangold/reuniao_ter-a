import numpy as np
import matplotlib.pyplot as plt
from   random import shuffle

class nn:
	def __init__(self, number_neurons):
		self.number_neurons = number_neurons

def activation(x: float) -> float:
	return 1/(1+np.exp(-x))

def cross_entropy_loss(predicted_distribution: list, true_distribution: list) -> float:
	total_loss = [true_distribution[index]*np.log(predicted_distribution[index]) 
			   	  for index in range(len(predicted_distribution))]
	
	return -np.sum(total_loss)/(2*len(total_loss))

def train(data, num_neurons_first_hidden_layer, num_neurons_second_hidden_layer):
	input_data = [[x[0], x[1]] for x in data]
	labels     = [x[2] for x in data]

	weight_matrix_first_layer  = np.random.rand(num_neurons_first_hidden_layer, len(input_data[0]))
	biases_first_layer 		   = np.random.rand(num_neurons_first_hidden_layer, 1)

	weight_matrix_second_layer = np.random.rand(num_neurons_second_hidden_layer, num_neurons_first_hidden_layer)
	biases_second_layer 	   = np.random.rand(num_neurons_second_hidden_layer, 1)
	for epoch in range(0, 2):
		for index_point in range(len(data)):
			first_hidden_layer_combination = np.add(weight_matrix_first_layer.dot(np.array(input_data[index_point])), biases_first_layer.T)
			first_hidden_layer_activated   = np.array([activation(neuron) for neuron in first_hidden_layer_combination])

			second_hidden_layer_combination = np.add(weight_matrix_second_layer.dot(np.array(first_hidden_layer_activated.T)), biases_second_layer)
			second_hidden_layer_activated   = np.array([np.exp(neuron)/sum(np.exp(second_hidden_layer_combination)) for neuron in second_hidden_layer_combination])

			error = cross_entropy_loss(second_hidden_layer_activated.T, [labels[index_point]])

			print(error)
			
training_linear    = [[x, x, [1, 0, 0]] for x in range(-200, 200)]			
training_quadratic = [[x, x**2, [0, 1, 0]] for x in range(-200, 200)]
training_cube      = [[x, x**3, [0, 0, 1]] for x in range(-200, 200)]

training_set = (training_quadratic + training_cube + training_linear)
shuffle(training_set)

#plt.plot(training_set)
#plt.show()

train(training_set, 6, 3)