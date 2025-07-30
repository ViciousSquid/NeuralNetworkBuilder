import os
import sys
import random
import matplotlib.pyplot as plt 

# Corrected path manipulation
grandparent_dir = os.path.dirname(os.path.abspath(__file__))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from NeuralNetwork.core import Network, Config
from NeuralNetwork.learning import BackpropNetwork

def main():
    print("XOR problem example using Backpropagation.")
    
    config = Config()
    config.hebbian['learning_interval'] = float('inf') 
    network = Network()  # Fixed: Instantiated Network properly
    network.set_neurogenesis_enabled(False)
    
    # Define network structure
    network.add_neuron("input1", value=0, position=(100, 100), n_type="input")
    network.add_neuron("input2", value=0, position=(100, 250), n_type="input")
    network.add_neuron("hidden1", value=50, position=(250, 100), n_type="hidden")
    network.add_neuron("hidden2", value=50, position=(250, 250), n_type="hidden")
    network.add_neuron("output1", value=50, position=(400, 175), n_type="output")
    
    # Create connections
    for input_name in ["input1", "input2"]:
        for hidden_name in ["hidden1", "hidden2"]:
            weight = random.uniform(-0.5, 0.5)
            network.connect(input_name, hidden_name, weight)
    for hidden_name in ["hidden1", "hidden2"]:
        weight = random.uniform(-0.5, 0.5)
        network.connect(hidden_name, "output1", weight)
    
    # Initialize the backpropagation learner
    backprop_learner = BackpropNetwork(network, learning_rate=0.7, momentum_factor=0.3)
    backprop_learner.set_layers([
        ["input1", "input2"],      
        ["hidden1", "hidden2"],    
        ["output1"]                
    ])
    
    # Training data
    training_data_normalized = [
        ([0.0, 0.0], [0.0]), ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]), ([1.0, 1.0], [0.0])
    ]
    
    print("\nTraining network on the XOR problem...")
    max_epochs = 5000 
    target_err = 0.005 
    
    # Training progress callback
    def training_progress_callback(epoch, avg_error):
        if epoch % 200 == 0 or epoch == max_epochs - 1:
            print(f"Epoch {epoch+1}/{max_epochs}: Avg Error = {avg_error:.7f}")
        return True

    epoch_errors_history = backprop_learner.train(
        training_data_normalized, 
        epochs=max_epochs, 
        target_error_threshold=target_err,
        progress_callback=training_progress_callback
    )
    
    print("\n--- Training Complete ---")
    
    # Test the trained network
    print("\nTesting trained network (normalized inputs/outputs):")
    correct_predictions = 0
    for inputs_norm, expected_norm in training_data_normalized:
        actual_outputs_norm = backprop_learner.forward_pass(inputs_norm)
        predicted_binary = 1 if actual_outputs_norm[0] > 0.5 else 0
        expected_binary = 1 if expected_norm[0] > 0.5 else 0
        print(f"  Input: {inputs_norm}, Raw Output: [{actual_outputs_norm[0]:.4f}], "
              f"Predicted Binary: {predicted_binary}, Expected Binary: {expected_binary}")
        if predicted_binary == expected_binary:
            correct_predictions += 1
            
    accuracy = (correct_predictions / len(training_data_normalized)) * 100
    print(f"\nAccuracy on training data: {accuracy:.2f}%")
    
    # Plot training error
    if epoch_errors_history:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_errors_history)
        plt.title('XOR Training Error (Backpropagation)')
        plt.xlabel('Epoch')
        plt.ylabel('Average Mean Squared Error')
        plt.grid(True)
        plt.show()
        print("Error plot displayed. You may need to close it to continue.")
    
    print("\nFinal connection weights after training:")
    if network.connections:
        for (src, tgt), conn_obj in sorted(network.connections.items()): 
            print(f"  {src} -> {tgt}: {conn_obj.get_weight():.4f}")
    else:
        print("  No connections in the network.")

if __name__ == "__main__":
    main()