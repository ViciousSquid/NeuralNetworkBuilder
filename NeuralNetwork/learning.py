# NeuralNetwork/learning.py
import math
import random

class BackpropNetwork:
    def __init__(self, network, learning_rate=0.5, momentum_factor=0.2):
        self.network = network
        self.learning_rate = learning_rate
        self.momentum = momentum_factor
        self.layers = []
        self.previous_weight_updates = {}

    def set_layers(self, layer_list):
        self.layers = layer_list

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def _sigmoid_derivative(self, y):
        return y * (1.0 - y)

    def forward_pass(self, inputs):
        # Set input neuron values
        input_layer = self.layers[0]
        for i, input_name in enumerate(input_layer):
            self.network.state[input_name] = inputs[i]
        
        # Propagate through the layers
        for i in range(1, len(self.layers)):
            prev_layer_names = self.layers[i-1]
            curr_layer_names = self.layers[i]
            
            for curr_neuron in curr_layer_names:
                net_input = 0
                for prev_neuron in prev_layer_names:
                    # Check for connection in both directions
                    conn = self.network.connections.get((prev_neuron, curr_neuron)) or \
                           self.network.connections.get((curr_neuron, prev_neuron))
                    if conn:
                        net_input += self.network.state[prev_neuron] * conn.get_weight()
                
                self.network.state[curr_neuron] = self._sigmoid(net_input)
                
        # Return output layer values
        return [self.network.state[name] for name in self.layers[-1]]

    def train(self, training_data, epochs=1000, target_error_threshold=0.01, progress_callback=None):
        epoch_errors = []
        for epoch in range(epochs):
            total_error = 0
            for inputs, expected_outputs in training_data:
                # Forward pass
                actual_outputs = self.forward_pass(inputs)
                
                # Calculate error and deltas for the output layer
                output_deltas = {}
                for i, output_name in enumerate(self.layers[-1]):
                    error = expected_outputs[i] - actual_outputs[i]
                    total_error += error**2
                    output_deltas[output_name] = error * self._sigmoid_derivative(actual_outputs[i])
                
                # Backpropagate error
                deltas = {**output_deltas}
                for i in range(len(self.layers) - 2, 0, -1):
                    hidden_deltas = {}
                    for hidden_name in self.layers[i]:
                        error = 0
                        for next_layer_name in self.layers[i+1]:
                            conn = self.network.connections.get((hidden_name, next_layer_name)) or \
                                   self.network.connections.get((next_layer_name, hidden_name))
                            if conn:
                                error += deltas[next_layer_name] * conn.get_weight()
                        hidden_deltas[hidden_name] = error * self._sigmoid_derivative(self.network.state[hidden_name])
                    deltas.update(hidden_deltas)
                
                # Update weights
                for i in range(1, len(self.layers)):
                    for prev_neuron in self.layers[i-1]:
                        for curr_neuron in self.layers[i]:
                            conn_key = (prev_neuron, curr_neuron)
                            rev_key = (curr_neuron, prev_neuron)
                            conn = self.network.connections.get(conn_key) or self.network.connections.get(rev_key)
                            
                            if conn:
                                momentum_term = self.momentum * self.previous_weight_updates.get(conn_key, 0)
                                weight_update = (self.learning_rate * deltas[curr_neuron] * self.network.state[prev_neuron]) + momentum_term
                                conn.set_weight(conn.get_weight() + weight_update)
                                self.previous_weight_updates[conn_key] = weight_update

            avg_error = total_error / len(training_data)
            epoch_errors.append(avg_error)
            
            if progress_callback and not progress_callback(epoch, avg_error):
                break
                
            if avg_error <= target_error_threshold:
                print(f"\nTarget error reached at epoch {epoch+1}")
                break
        
        return epoch_errors