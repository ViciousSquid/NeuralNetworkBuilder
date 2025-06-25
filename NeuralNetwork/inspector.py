# NeuralNetwork/inspector.py
from PyQt5 import QtWidgets, QtCore

class NeuronInspectorDialog(QtWidgets.QDialog):
    neuronPropertyChanged = QtCore.pyqtSignal(str, str, object)

    def __init__(self, neuron_name, network, parent=None):
        super().__init__(parent)
        self.neuron_name = neuron_name
        self.network = network
        self.setWindowTitle(f"Inspector: {neuron_name}")
        self.setup_ui()
        self.populate_all_data()

    def setup_ui(self):
        layout = QtWidgets.QFormLayout(self)
        self.name_edit = QtWidgets.QLineEdit(self.neuron_name)
        self.type_label = QtWidgets.QLabel()
        self.value_spin = QtWidgets.QDoubleSpinBox()
        self.value_spin.setRange(-100, 100)
        
        layout.addRow("Name:", self.name_edit)
        layout.addRow("Type:", self.type_label)
        layout.addRow("Activation:", self.value_spin)

        self.connections_table = QtWidgets.QTableWidget()
        self.connections_table.setColumnCount(2)
        self.connections_table.setHorizontalHeaderLabels(["Target", "Weight"])
        layout.addRow(QtWidgets.QLabel("Outgoing Connections:"))
        layout.addRow(self.connections_table)

        # Connect signals
        self.name_edit.editingFinished.connect(self.on_name_changed)
        self.value_spin.valueChanged.connect(self.on_value_changed)

    def on_name_changed(self):
        new_name = self.name_edit.text()
        if new_name != self.neuron_name:
            self.neuronPropertyChanged.emit(self.neuron_name, "name", new_name)
    
    def on_value_changed(self, value):
        self.neuronPropertyChanged.emit(self.neuron_name, "state_value", value)
        
    def populate_all_data(self):
        neuron = self.network.neurons.get(self.neuron_name)
        if not neuron:
            self.close()
            return

        self.name_edit.setText(neuron.name)
        self.type_label.setText(neuron.type)
        self.value_spin.setValue(self.network.state.get(self.neuron_name, 0))
        self.populate_connections_tab()
    
    def populate_connections_tab(self):
        self.connections_table.setRowCount(0)
        outgoing = []
        for (s, t), conn in self.network.connections.items():
            if s == self.neuron_name:
                outgoing.append((t, conn.get_weight()))

        self.connections_table.setRowCount(len(outgoing))
        for row, (target, weight) in enumerate(outgoing):
            self.connections_table.setItem(row, 0, QtWidgets.QTableWidgetItem(target))
            self.connections_table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{weight:.3f}"))
            
    def update_neuron_reference(self, new_name):
        self.neuron_name = new_name
        self.setWindowTitle(f"Inspector: {new_name}")
        self.populate_all_data()