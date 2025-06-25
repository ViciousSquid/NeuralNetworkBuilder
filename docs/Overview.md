# Hebbian Learning
The implementation of Hebbian learning follows the principle "neurons that fire together, wire together."

### Implementation (core.py): The perform_learning method in the Network class contains the core logic. It identifies pairs of neurons whose activation levels are simultaneously above the "Active Threshold" and increases the connection weight between them. A weight decay mechanism is also included to gradually weaken unused connections, promoting network efficiency.
  
### Trigger Mechanism:
* Manual Trigger: The "Perform Hebbian Learning" button in the GUI allows the user to force a learning cycle at any moment.
* Automatic Interval: The perform_learning method is also designed to be triggered automatically. It checks the time elapsed since its last execution and runs if the time exceeds the "Hebbian Interval" set in the UI's "Learning Parameters" section. This creates a continuous, unsupervised learning process in the background without needing a visible countdown.

-----------------------------------

# Neurogenesis
Neurogenesis, the creation of new neurons, is implemented as a response to specific, high-level states within the network, simulating an environment's effect on brain structure. The GUI does not generate these states itself; it provides the framework to respond to them.

### Implementation (core.py): The check_neurogenesis method in the Network class is the heart of this feature. It is not triggered by a simple timer but by the state of specific variables passed into it.

* The "Hypothetical Simulation" Trigger:
The system is designed to be driven by a larger, hypothetical simulation that would manage the network's "environment." This external simulation would be responsible for updating special state variables within the network to represent abstract conditions. The code reveals these trigger variables:

* SIM_novelty_exposure: Represents the network being exposed to a new, unfamiliar stimulus.
* SIM_sustained_stress: Represents a prolonged period of adverse conditions.
* SIM_recent_rewards: Represents positive feedback or achieving a goal.
The check_neurogenesis function evaluates the values of these variables against configurable thresholds. If a threshold is crossed (e.g., if SIM_novelty_exposure is very high), a new neuron is created, typically of a corresponding type like "novelty" or "stress".

The "Trigger Neurogenesis" button in the GUI serves as a debugging tool to demonstrate this. When clicked, it simulates this external trigger by temporarily creating a sim_state_for_neuro dictionary with a high novelty value, thereby forcing a neurogenesis event for demonstration.

-----------------------------------

# Supervised Learning: Backpropagation

For tasks that require supervised learning, the project includes a BackpropNetwork class.

### Purpose: This class implements the backpropagation algorithm, which adjusts connection weights to minimize the difference between the network's output and a known correct output.
* Usage: It requires the network's layers to be explicitly defined. The train method takes a dataset of input-output pairs and iterates through them for a number of epochs, progressively reducing the error. This is demonstrated in several of the example files:
* backprop_xor.py: Solves the classic XOR logic problem.
* pong_ai.py: Trains a simple AI to play Pong by learning from a "perfect" algorithm.
* webcam_color_recognition.py: Trains a network to recognize colors from a live webcam feed based on user-provided samples.

-----------------------------------

 ### Examples
The examples directory showcases the versatility of the framework:

* basic_network.py: A command-line script that demonstrates the core concepts of activation propagation, Hebbian learning, and neurogenesis without a GUI.
* flocking_boids.py: A visual simulation of flocking behavior (like birds or fish) where each "boid" is controlled by its own simple, hand-tuned neural network.
* visualization_example.py: A minimal GUI application that demonstrates how to use the NetworkVisualization widget and periodically stimulates the network to show dynamic activity.
