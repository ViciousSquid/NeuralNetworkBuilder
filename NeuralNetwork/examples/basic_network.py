import os
import sys
import time
import random

# Add the project root to sys.path
grandparent_dir = os.path.dirname(os.path.abspath(__file__))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from NeuralNetwork.core import Network, Config

def main():
    # Create a network with default configuration
    config = Config()
    # For demonstration, let's make learning and neurogenesis happen more frequently
    config.hebbian['learning_interval'] = 1000  # 1 second
    config.neurogenesis['cooldown'] = 3     # 3 seconds
    config.neurogenesis['novelty_threshold'] = 2.0 # Lower threshold for demo
    
    network = Network(config)
    network.set_neurogenesis_enabled(True) # Explicitly enable neurogenesis

    print("Creating a simple network with Hebbian learning and Neurogenesis...")

    # Add neurons
    network.add_neuron("input1", value=0, position=(100, 100), neuron_type="input")
    network.add_neuron("input2", value=0, position=(100, 250), neuron_type="input")
    network.add_neuron("hidden1", value=0, position=(250, 175), neuron_type="hidden")
    network.add_neuron("output1", value=0, position=(400, 175), neuron_type="output")

    # Connect neurons
    network.connect("input1", "hidden1", weight=random.uniform(-0.5, 0.5))
    network.connect("input2", "hidden1", weight=random.uniform(-0.5, 0.5))
    network.connect("hidden1", "output1", weight=random.uniform(-0.5, 0.5))

    print("Network structure created.")
    print(f"Neurons: {len(network.neurons)}")
    print(f"Connections: {len(network.connections)}")

    # Simulation loop
    print("\nRunning simulation for 10 iterations...")
    for i in range(10):
        # Stimulate input neurons
        input1_val = random.uniform(0, 100) 
        input2_val = random.uniform(0, 100)

        network.state.update({
            "input1": input1_val,
            "input2": input2_val
        })

        print(f"\nIteration {i+1}:")
        print(f"  Input1: {network.state.get('input1', 0):.2f}, Input2: {network.state.get('input2', 0):.2f}")

        # Propagate activation multiple times
        for _ in range(3): 
            network.propagate_activation()

        hidden1_val = network.state.get("hidden1", 0)
        output1_val = network.state.get("output1", 0)

        print(f"  Hidden1: {hidden1_val:.2f}")
        print(f"  Output1: {output1_val:.2f}")

        # Perform Hebbian learning
        updated_pairs = network.perform_learning()
        if updated_pairs is not None: 
            if updated_pairs:
                print(f"  Learning: Updated {len(updated_pairs)} connection pair(s).")
            else:
                print("  Learning: No significant co-activity for updates.")
        else:
            print("  Learning: Skipped (too soon).")

        # Check for neurogenesis
        if i % 3 == 0: 
            simulated_state_for_neurogenesis = {
                'SIM_novelty_exposure': random.uniform(1.0, config.neurogenesis['novelty_threshold'] * 1.5),
                'SIM_sustained_stress': random.uniform(0.0, config.neurogenesis['stress_threshold'] * 0.5), 
                'SIM_recent_rewards': random.uniform(0.0, config.neurogenesis['reward_threshold'] * 0.5)
            }
            full_state_for_check = {**network.state, **simulated_state_for_neurogenesis}

            new_neuron_name = network.check_neurogenesis(full_state_for_check)
            if new_neuron_name:
                print(f"  Neurogenesis: Created new neuron '{new_neuron_name}'!")
                print(f"  Network now has {len(network.neurons)} neurons.")
            else:
                status = "disabled" if not network.neurogenesis_enabled else "conditions not met or on cooldown"
                print(f"  Neurogenesis: No new neurons created ({status}).")
        
        time.sleep(0.5) 

    print("\n--- Simulation Ended ---")
    print("Final neuron states:")
    for name, value in network.state.items():
        if name in network.neurons:
             print(f"  {name}: {value:.2f}")

    print("\nFinal connection weights:")
    if network.connections:
        for (src, tgt), conn_obj in network.connections.items():
            print(f"  {src} -> {tgt}: {conn_obj.get_weight():.4f}")
    else:
        print("  No connections in the network.")

if __name__ == "__main__":
    main()