import sys
import os
import time
import random

# Corrected path manipulation
grandparent_dir = os.path.dirname(os.path.abspath(__file__))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from PyQt5 import QtWidgets, QtCore, QtGui 
from NeuralNetwork.core import Network, Config
from NeuralNetwork.visualization import NetworkVisualization

class NetworkVisApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neural Network Visualization Example")
        self.resize(1000, 750)
        
        self.network = self.create_sample_network()
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        
        self.vis = NetworkVisualization(self.network, self) 
        self.vis.set_layers_data({}) 
        main_layout.addWidget(self.vis)
        
        controls_group = QtWidgets.QGroupBox("Controls")
        controls_layout = QtWidgets.QGridLayout(controls_group)

        self.show_weights_cb = QtWidgets.QCheckBox("Show Weights")
        self.show_weights_cb.setChecked(self.vis.show_weights)
        self.show_weights_cb.toggled.connect(self.toggle_weights_display)
        controls_layout.addWidget(self.show_weights_cb, 0, 0)

        self.show_links_cb = QtWidgets.QCheckBox("Show Links")
        self.show_links_cb.setChecked(self.vis.show_links)
        self.show_links_cb.toggled.connect(self.toggle_links_display)
        controls_layout.addWidget(self.show_links_cb, 0, 1)
        
        self.perform_learning_btn = QtWidgets.QPushButton("Perform Learning")
        self.perform_learning_btn.clicked.connect(self.run_learning_cycle)
        controls_layout.addWidget(self.perform_learning_btn, 1, 0)
        
        self.stimulate_random_btn = QtWidgets.QPushButton("Stimulate Randomly")
        self.stimulate_random_btn.clicked.connect(self.run_random_stimulation)
        controls_layout.addWidget(self.stimulate_random_btn, 1, 1)
        
        self.add_random_neuron_btn = QtWidgets.QPushButton("Add Random Neuron")
        self.add_random_neuron_btn.clicked.connect(self.add_new_random_neuron)
        controls_layout.addWidget(self.add_random_neuron_btn, 2, 0)

        self.propagate_btn = QtWidgets.QPushButton("Propagate Activation")
        self.propagate_btn.clicked.connect(self.run_propagation)
        controls_layout.addWidget(self.propagate_btn, 2, 1)
        
        main_layout.addWidget(controls_group)
        
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.periodic_network_update) 
        self.update_timer.start(2000)
        
        self.vis.neuronClicked.connect(self.display_neuron_info)
    
    def create_sample_network(self):
        config = Config()
        config.hebbian['learning_interval'] = 100 
        config.neurogenesis['cooldown'] = 5
        # This line has a separate error; it should be corrected as per previous instructions.
        network = Network() # Corrected from Network(config)
        network.config = config # Added this line
        network.set_neurogenesis_enabled(True)

        neuron_definitions = [
            ("N1_In", "input", (100, 100)), ("N2_In", "input", (100, 250)),
            ("H1", "hidden", (300, 100)), ("H2", "hidden", (300, 250)),
            ("O1_Out", "output", (500, 175))
        ]
        for name, n_type, pos in neuron_definitions:
            # Corrected keyword from 'neuron_type' to 'n_type'
            network.add_neuron(name, value=random.uniform(0,100), position=pos, n_type=n_type)

        connections_to_make = [
            ("N1_In", "H1"), ("N1_In", "H2"),
            ("N2_In", "H1"), ("N2_In", "H2"),
            ("H1", "O1_Out"), ("H2", "O1_Out")
        ]
        for src, tgt in connections_to_make:
            network.connect(src, tgt, weight=random.uniform(-0.6, 0.6))
        return network
    
    def toggle_weights_display(self, checked):
        self.vis.show_weights = checked
        self.vis.update()
    
    def toggle_links_display(self, checked):
        self.vis.show_links = checked
        self.vis.update()
    
    def run_learning_cycle(self):
        updated_pairs = self.network.perform_learning()
        if updated_pairs is not None:
            msg = f"Learning: {len(updated_pairs)} pairs updated." if updated_pairs else "Learning: No co-activity."
            self.statusBar().showMessage(msg, 3000)
        else:
            self.statusBar().showMessage("Learning: Skipped (too soon).", 3000)
        self.vis.update()
    
    def run_random_stimulation(self):
        if not self.network.neurons: return
        changes = {}
        for name, neuron in self.network.neurons.items():
            if neuron.type == "input":
                changes[name] = random.uniform(60, 100)
            elif random.random() < 0.3: 
                changes[name] = random.uniform(0, 50)
        if changes:
            self.network.state.update(changes)
            for name in changes:
                # Corrected method name and removed extra argument
                self.vis.highlight_new_neuron(name, 0.7)
            self.statusBar().showMessage(f"Stimulated {len(changes)} neurons.", 3000)
            self.vis.update()

    def add_new_random_neuron(self):
        idx = len(self.network.neurons)
        new_name = f"RndN_{idx}"
        logical_center_x = -self.vis.pan_offset.x() + (self.vis.width() / self.vis.zoom_factor / 2)
        logical_center_y = -self.vis.pan_offset.y() + (self.vis.height() / self.vis.zoom_factor / 2)
        
        pos = (logical_center_x + random.randint(-100, 100),
            logical_center_y + random.randint(-100, 100))
        
        neuron_types = ["default", "hidden"]
        n_type = random.choice(neuron_types)
        
        # Corrected keyword from 'neuron_type' to 'n_type'
        self.network.add_neuron(new_name, value=50.0, position=pos, n_type=n_type)
        
        num_to_connect = min(2, len(self.network.neurons) -1)
        if num_to_connect > 0:
            existing_neurons = [n for n in self.network.neurons if n != new_name]
            for target in random.sample(existing_neurons, k=num_to_connect):
                self.network.connect(new_name, target, random.uniform(-0.3, 0.3))
                self.network.connect(target, new_name, random.uniform(-0.3, 0.3))

        self.vis.highlight_new_neuron(new_name, 3.0)
        self.statusBar().showMessage(f"Added neuron: {new_name}", 3000)
        self.vis.update()

    def run_propagation(self):
        self.network.propagate_activation()
        for name in self.network.neurons:
            # Corrected method name and removed extra argument
            self.vis.highlight_new_neuron(name, 0.5)
        self.statusBar().showMessage("Activation propagated.", 3000)
        self.vis.update()
        
    def periodic_network_update(self):
        if not self.network.neurons: return
        for name in random.sample(list(self.network.neurons.keys()), k=max(1, len(self.network.neurons)//5)):
            current_val = self.network.state.get(name, 50.0)
            drift = random.uniform(-5, 5)
            self.network.state.update({name: max(0, min(100, current_val + drift))})
        
        if random.random() < 0.2: 
            sim_state = {
                'SIM_novelty_exposure': random.uniform(0, self.network.config.neurogenesis['novelty_threshold'] * 1.2),
            }
            new_n = self.network.check_neurogenesis({**self.network.state, **sim_state})
            if new_n:
                self.vis.highlight_new_neuron(new_n, 5.0)
                self.statusBar().showMessage(f"Neurogenesis: '{new_n}' created.", 3000)
        self.vis.update()
    
    def display_neuron_info(self, neuron_name):
        if neuron_name in self.network.neurons:
            neuron = self.network.neurons[neuron_name]
            state_val = self.network.state.get(neuron_name, 0.0)
            num_outgoing = sum(1 for (src, _) in self.network.connections if src == neuron_name)
            num_incoming = sum(1 for (_, tgt) in self.network.connections if tgt == neuron_name)
            info_text = (f"<b>Neuron: {neuron.name}</b><br>"
                         f"Type: {neuron.type}<br>"
                         f"Activation: {state_val:.2f}<br>"
                         f"Position: ({neuron.position[0]:.0f}, {neuron.position[1]:.0f})<br>"
                         f"Outgoing Connections: {num_outgoing}<br>"
                         f"Incoming Connections: {num_incoming}")
            QtWidgets.QMessageBox.information(self, "Neuron Info", info_text)

    def closeEvent(self, event):
        self.update_timer.stop()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    if "Fusion" in QtWidgets.QStyleFactory.keys():
        app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    window = NetworkVisApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()