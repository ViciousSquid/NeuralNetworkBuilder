import sys
import os
import random
import time
from PyQt5 import QtWidgets, QtCore, QtGui
import json
import math

# Adapted from the original project to use the new modular structure
from NeuralNetwork.core import Network, Config
from NeuralNetwork.visualization import NetworkVisualization
from NeuralNetwork.inspector import NeuronInspectorDialog

class NetworkBuilderGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.network = Network()
        self.layers = {}
        self.active_inspectors = {}
        self._define_context_menu_handlers()
        self.setup_ui()
        self.mode = "select"
        self.selected_item = None
        self.connection_start_neuron = None
        self.neuron_counter = 0
        self.layer_counter = 0
        self.set_mode("select")

    def _define_context_menu_handlers(self):
        def show_visualization_context_menu(position_widget):
            clicked_neuron_name = self.vis.get_neuron_at_pos(position_widget)
            menu = QtWidgets.QMenu(self)
            if clicked_neuron_name:
                menu.addAction(f"Inspect: {clicked_neuron_name}", lambda: self.open_neuron_inspector(clicked_neuron_name))
                menu.addSeparator()
                menu.addAction(f"Remove: {clicked_neuron_name}", lambda: self.handle_remove_neuron_action(clicked_neuron_name))
            else:
                clicked_layer_name = self.vis.get_layer_at_pos(position_widget)
                if clicked_layer_name:
                    menu.addAction(f"Layer: {clicked_layer_name} (Selected)", lambda l=clicked_layer_name: self.select_layer_action(l))
                    menu.addSeparator()
            if self.vis.selected_neurons:
                menu.addAction(f"Clear {len(self.vis.selected_neurons)} Selection(s)", self.clear_selection_action)
            menu.addAction("Refresh View", self.vis.update)
            menu.exec_(self.vis.mapToGlobal(position_widget))
        self.show_visualization_context_menu = show_visualization_context_menu

    def setup_ui(self):
        self.setWindowTitle("Neural Network Builder & Visualizer v1.1")
        self.setGeometry(100, 100, 1500, 950)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)
        self.control_panel = QtWidgets.QWidget()
        self.control_panel.setMaximumWidth(380)
        control_layout = QtWidgets.QVBoxLayout(self.control_panel)
        self.create_tools_group(control_layout)
        self.create_layer_tools_group(control_layout)
        self.create_properties_panel(control_layout)
        self.create_simulation_controls(control_layout)
        control_layout.addStretch()
        self.vis_panel = QtWidgets.QWidget()
        vis_layout = QtWidgets.QVBoxLayout(self.vis_panel)
        self.vis = NetworkVisualization(self.network, self)
        self.vis.set_layers_data(self.layers)

        self.vis.canvasClicked.connect(self.visualization_mouse_press)
        self.vis.canvasMoved.connect(self.visualization_mouse_move)
        self.vis.canvasReleased.connect(self.visualization_mouse_release)
        
        self.vis.neuronClicked.connect(self.on_neuron_vis_clicked_for_property_panel)
        self.vis.selectionChanged.connect(self.on_selection_changed_gui)
        self.vis.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.vis.customContextMenuRequested.connect(self.show_visualization_context_menu)
        
        vis_layout.addWidget(self.vis)
        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_label = QtWidgets.QLabel("Zoom:")
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom_slider.setRange(10, 400); self.zoom_slider.setValue(100)
        self.zoom_slider.setTickInterval(25); self.zoom_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_layout.addWidget(zoom_label); zoom_layout.addWidget(self.zoom_slider)
        vis_layout.addLayout(zoom_layout)
        self.splitter.addWidget(self.control_panel); self.splitter.addWidget(self.vis_panel)
        self.splitter.setSizes([350, 1150])
        self.create_menu_bar()
        self.stats_label = QtWidgets.QLabel("Neurons: 0 | Connections: 0")
        self.statusBar().addPermanentWidget(self.stats_label)
        self.update_network_statistics()

    def on_zoom_changed(self, value):
        self.vis.set_zoom(value / 100.0)

    def create_tools_group(self, layout):
        group = QtWidgets.QGroupBox("Tools")
        group_layout = QtWidgets.QGridLayout(group)
        self.select_btn = QtWidgets.QPushButton(QtGui.QIcon.fromTheme("edit-select-all"), " Select")
        self.add_neuron_btn = QtWidgets.QPushButton(QtGui.QIcon.fromTheme("list-add"), " Add Neuron")
        self.add_connection_btn = QtWidgets.QPushButton(QtGui.QIcon.fromTheme("draw-connector"), " Add Connection")
        self.remove_btn = QtWidgets.QPushButton(QtGui.QIcon.fromTheme("edit-delete"), " Remove")
        self.mode_buttons = [self.select_btn, self.add_neuron_btn, self.add_connection_btn, self.remove_btn]
        modes_defs = [("select", self.select_btn, "Select items. Drag to move. Shift+Click for multi-select. Rubber-band select."),
                      ("add_neuron", self.add_neuron_btn, "Click canvas to add a neuron."),
                      ("add_connection", self.add_connection_btn, "Drag between neurons to connect."),
                      ("remove", self.remove_btn, "Click item to remove.")]
        for i, (mode, btn, tip) in enumerate(modes_defs):
            btn.setCheckable(True); btn.setToolTip(tip)
            btn.clicked.connect(lambda c, m=mode: self.set_mode(m))
            group_layout.addWidget(btn, i // 2, i % 2)
        self.neuron_type_combo_adder = QtWidgets.QComboBox()
        self.neuron_type_combo_adder.addItems(["default", "input", "hidden", "output", "novelty", "stress", "reward"])
        self.neuron_type_combo_adder.setToolTip("Type for new neurons.")
        group_layout.addWidget(QtWidgets.QLabel("New Neuron Type:"), 2, 0, 1, 2)
        group_layout.addWidget(self.neuron_type_combo_adder, 3, 0, 1, 2)
        layout.addWidget(group)

    def create_layer_tools_group(self, layout):
        group = QtWidgets.QGroupBox("Layer & Layout Tools")
        group_layout = QtWidgets.QVBoxLayout(group)
        btn_defs = [("Add Layer", self.add_layer_dialog, "Add a new layer."),
                    ("Connect Layers", self.connect_layers_dialog, "Connect two layers."),
                    ("Create Feedforward Network", self.create_feedforward_dialog, "Create a new feedforward network."),
                    ("Auto-Layout Network", self.auto_layout_network, "Arrange neurons automatically (experimental).")]
        for txt, func, tip in btn_defs:
            btn = QtWidgets.QPushButton(txt); btn.clicked.connect(func); btn.setToolTip(tip)
            group_layout.addWidget(btn)
        layout.addWidget(group)

    def create_properties_panel(self, layout):
        group = QtWidgets.QGroupBox("Properties")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.neuron_props_widget = QtWidgets.QWidget()
        n_form = QtWidgets.QFormLayout(self.neuron_props_widget)
        self.neuron_name_edit = QtWidgets.QLineEdit(); self.neuron_name_edit.editingFinished.connect(self.update_neuron_property_from_panel)
        self.neuron_value_spin = QtWidgets.QDoubleSpinBox(); self.neuron_value_spin.setRange(-1000,1000); self.neuron_value_spin.setDecimals(3); self.neuron_value_spin.valueChanged.connect(self.update_neuron_property_from_panel)
        self.neuron_type_props_combo = QtWidgets.QComboBox(); self.neuron_type_props_combo.addItems(["default","input","hidden","output","novelty","stress","reward"]); self.neuron_type_props_combo.currentTextChanged.connect(self.update_neuron_property_from_panel)
        self.neuron_x_spin = QtWidgets.QSpinBox(); self.neuron_x_spin.setRange(-10000,10000); self.neuron_x_spin.valueChanged.connect(self.update_neuron_property_from_panel)
        self.neuron_y_spin = QtWidgets.QSpinBox(); self.neuron_y_spin.setRange(-10000,10000); self.neuron_y_spin.valueChanged.connect(self.update_neuron_property_from_panel)
        n_form.addRow("Name:",self.neuron_name_edit); n_form.addRow("Activation:",self.neuron_value_spin)
        n_form.addRow("Type:",self.neuron_type_props_combo); n_form.addRow("X:",self.neuron_x_spin); n_form.addRow("Y:",self.neuron_y_spin)
        self.connection_props_widget = QtWidgets.QWidget()
        c_form = QtWidgets.QFormLayout(self.connection_props_widget)
        self.connection_source_label = QtWidgets.QLabel(); self.connection_target_label = QtWidgets.QLabel()
        self.connection_weight_spin = QtWidgets.QDoubleSpinBox(); self.connection_weight_spin.setRange(-1,1); self.connection_weight_spin.setSingleStep(0.01); self.connection_weight_spin.setDecimals(3); self.connection_weight_spin.valueChanged.connect(self.update_connection_property_from_panel)
        c_form.addRow("Source:",self.connection_source_label); c_form.addRow("Target:",self.connection_target_label); c_form.addRow("Weight:",self.connection_weight_spin)
        self.multi_select_label = QtWidgets.QLabel("Multiple items selected."); self.multi_select_label.setAlignment(QtCore.Qt.AlignCenter); self.multi_select_label.setWordWrap(True)
        self.nothing_selected_label = QtWidgets.QLabel("No item selected."); self.nothing_selected_label.setAlignment(QtCore.Qt.AlignCenter)
        self.properties_stack = QtWidgets.QStackedWidget()
        for w in [self.neuron_props_widget, self.connection_props_widget, self.multi_select_label, self.nothing_selected_label]: self.properties_stack.addWidget(w)
        self.properties_stack.setCurrentIndex(3)
        group_layout.addWidget(self.properties_stack)
        layout.addWidget(group)

    def create_simulation_controls(self, layout):
        group = QtWidgets.QGroupBox("Simulation & Learning")
        group_layout = QtWidgets.QVBoxLayout(group)
        update_state_layout = QtWidgets.QHBoxLayout()
        self.update_neuron_combo = QtWidgets.QComboBox(); self.update_neuron_combo.addItem("Select Neuron"); self.update_neuron_combo.setToolTip("Select neuron to set activation.")
        self.update_value_spin = QtWidgets.QDoubleSpinBox(); self.update_value_spin.setRange(-1000,1000); self.update_value_spin.setDecimals(3); self.update_value_spin.setToolTip("Activation value.")
        self.update_state_btn = QtWidgets.QPushButton("Set Value"); self.update_state_btn.clicked.connect(self.update_single_neuron_state_from_controls); self.update_state_btn.setToolTip("Apply activation.")
        for w in [self.update_neuron_combo, self.update_value_spin, self.update_state_btn]: update_state_layout.addWidget(w)
        group_layout.addLayout(update_state_layout)
        self.perform_learning_btn = QtWidgets.QPushButton("Perform Hebbian Learning"); self.perform_learning_btn.clicked.connect(self.perform_learning_action); self.perform_learning_btn.setToolTip("Run Hebbian learning cycle.")
        group_layout.addWidget(self.perform_learning_btn)
        self.propagate_btn = QtWidgets.QPushButton("Propagate Activation"); self.propagate_btn.clicked.connect(self.propagate_activation_action); self.propagate_btn.setToolTip("Run activation propagation.")
        group_layout.addWidget(self.propagate_btn)
        
        neuro_group = QtWidgets.QGroupBox("Neurogenesis")
        neuro_layout = QtWidgets.QFormLayout(neuro_group)
        self.neurogenesis_enable_cb = QtWidgets.QCheckBox("Enable Neurogenesis")
        self.neurogenesis_enable_cb.setChecked(self.network.neurogenesis_enabled)
        self.neurogenesis_enable_cb.toggled.connect(self.network.set_neurogenesis_enabled) 
        self.neurogenesis_enable_cb.setToolTip("Globally enable or disable neurogenesis.")
        neuro_layout.addRow(self.neurogenesis_enable_cb)
        self.trigger_neuro_btn = QtWidgets.QPushButton("Trigger Neurogenesis")
        self.trigger_neuro_btn.clicked.connect(self.manually_trigger_neurogenesis_check)
        self.trigger_neuro_btn.setToolTip("Force a check for neurogenesis conditions with current (or simulated) values.")
        neuro_layout.addRow(self.trigger_neuro_btn)
        group_layout.addWidget(neuro_group)

        params_group = QtWidgets.QGroupBox("Learning Parameters")
        params_layout = QtWidgets.QFormLayout(params_group)
        self.lr_spin = QtWidgets.QDoubleSpinBox()
        self.lr_spin.setRange(0.0001,1.0); self.lr_spin.setSingleStep(0.001); self.lr_spin.setDecimals(4)
        self.lr_spin.setValue(self.network.config.hebbian.get('base_learning_rate',0.1))
        self.lr_spin.valueChanged.connect(lambda v:self.network.config.hebbian.update({'base_learning_rate':v}))
        params_layout.addRow("Learning Rate:",self.lr_spin)
        self.active_thresh_spin = QtWidgets.QSpinBox()
        self.active_thresh_spin.setRange(0,100)
        self.active_thresh_spin.setValue(self.network.config.hebbian.get('active_threshold',50))
        self.active_thresh_spin.valueChanged.connect(lambda v:self.network.config.hebbian.update({'active_threshold':v}))
        params_layout.addRow("Active Threshold:",self.active_thresh_spin)
        self.hebbian_interval_spin = QtWidgets.QSpinBox()
        self.hebbian_interval_spin.setRange(1000,300000); self.hebbian_interval_spin.setSingleStep(1000); self.hebbian_interval_spin.setSuffix(" ms")
        self.hebbian_interval_spin.setValue(self.network.config.hebbian.get('learning_interval',30000))
        self.hebbian_interval_spin.valueChanged.connect(lambda v:self.network.config.hebbian.update({'learning_interval':v}))
        params_layout.addRow("Hebbian Interval:",self.hebbian_interval_spin)
        group_layout.addWidget(params_group)
        layout.addWidget(group)

    def manually_trigger_neurogenesis_check(self):
        if not self.network.neurogenesis_enabled:
            self.statusBar().showMessage("Neurogenesis is disabled. Cannot trigger check.")
            return

        sim_state_for_neuro = {
            'SIM_novelty_exposure': self.network.config.neurogenesis['novelty_threshold'] * 1.1,
            'SIM_sustained_stress': 0, 
            'SIM_recent_rewards': 0
        }
        full_state_for_check = {**self.network.state, **sim_state_for_neuro}

        new_neuron_name = self.network.check_neurogenesis(full_state_for_check)
        if new_neuron_name:
            self.statusBar().showMessage(f"Manual Neurogenesis Check: Neuron '{new_neuron_name}' created!")
            self.update_simulation_combo()
            self.vis.highlight_new_neuron(new_neuron_name, 5.0)
            self.update_network_statistics()
            self.vis.update()
        else:
            self.statusBar().showMessage("Manual Neurogenesis Check: Conditions not met or cooldown active.")


    def create_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_actions = {"&New Network": (self.new_network_action, "Ctrl+N", "Create a new, empty network."), "&Open Network...": (self.open_network_action, "Ctrl+O", "Open a network from a JSON file."), "&Save Network...": (self.save_network_action, "Ctrl+S", "Save the current network to a JSON file."), "E&xit": (self.close, "Ctrl+Q", "Exit the application.")}
        for text, (func, shortcut, tooltip) in file_actions.items():
            action = QtWidgets.QAction(text, self); action.triggered.connect(func); action.setShortcut(shortcut); action.setStatusTip(tooltip)
            file_menu.addAction(action)
            if text == "&Save Network...": file_menu.addSeparator()
        edit_menu = menu_bar.addMenu("&Edit")
        edit_actions = {"&Clear Network": (self.clear_network_action, "Ctrl+Shift+N", "Remove all neurons and connections."), "&Randomize Weights": (self.randomize_weights_action, "", "Assign random weights to all connections.")}
        for text, (func, shortcut, tooltip) in edit_actions.items():
            action = QtWidgets.QAction(text, self); action.triggered.connect(func); action.setShortcut(shortcut); action.setStatusTip(tooltip)
            edit_menu.addAction(action)
        view_menu = menu_bar.addMenu("&View")
        self.show_weights_action = QtWidgets.QAction("Show Connection &Weights", self, checkable=True, checked=True); self.show_weights_action.triggered.connect(lambda c: setattr(self.vis, 'show_weights', c) or self.vis.update()); self.show_weights_action.setStatusTip("Toggle visibility of connection weight labels.")
        view_menu.addAction(self.show_weights_action)
        self.show_links_action = QtWidgets.QAction("Show Connection &Links", self, checkable=True, checked=True); self.show_links_action.triggered.connect(lambda c: setattr(self.vis, 'show_links', c) or self.vis.update()); self.show_links_action.setStatusTip("Toggle visibility of connection lines.")
        view_menu.addAction(self.show_links_action)
        help_menu = menu_bar.addMenu("&Help")
        about_action = QtWidgets.QAction("&About", self); about_action.triggered.connect(self.show_about_dialog); about_action.setStatusTip("Show application information.")
        help_menu.addAction(about_action)

    def set_mode(self, mode_name):
        self.mode = mode_name
        self.vis.current_mouse_mode = mode_name
        self.connection_start_neuron = None
        for btn in self.mode_buttons: btn.setChecked(False)
        mode_map = {"select": (self.select_btn, "Select Mode: Click or drag to select. Right-click/drag to pan. Wheel to zoom."), "add_neuron": (self.add_neuron_btn, "Add Neuron Mode: Click on canvas to add a neuron."), "add_connection": (self.add_connection_btn, "Add Connection Mode: Click and drag between two neurons."), "remove": (self.remove_btn, "Remove Mode: Click on a neuron or connection to delete.")}
        if mode_name in mode_map:
            btn_to_check, status_msg = mode_map[mode_name]
            btn_to_check.setChecked(True)
            self.statusBar().showMessage(status_msg)
        self.vis.update()

    def clear_selection_action(self):
        self.selected_item = None
        self.vis.selected_neurons.clear()
        self.on_selection_changed_gui(set())
        self.vis.update()
        
    def select_layer_action(self, layer_name):
        if layer_name in self.layers:
            self.vis.selected_neurons = set(self.layers[layer_name]['neurons'])
            self.on_selection_changed_gui(self.vis.selected_neurons)
            self.statusBar().showMessage(f"Selected all neurons in layer: {layer_name}")
            self.vis.update()

    def on_neuron_vis_clicked_for_property_panel(self, neuron_name):
        if self.mode == "select":
            modifiers = QtWidgets.QApplication.keyboardModifiers()
            is_shift = bool(modifiers & QtCore.Qt.ShiftModifier)
            if not is_shift:
                if neuron_name in self.vis.selected_neurons and len(self.vis.selected_neurons) > 1:
                    self.vis.selected_neurons = {neuron_name}
                    self.vis.selectionChanged.emit(self.vis.selected_neurons) 
            if len(self.vis.selected_neurons) == 1:
                 self.selected_item = list(self.vis.selected_neurons)[0]
                 self.update_property_panel()
            elif not self.vis.selected_neurons:
                 self.selected_item = None
                 self.update_property_panel()

    def on_selection_changed_gui(self, selected_neuron_names_set):
        if not selected_neuron_names_set:
            self.selected_item = None
            self.properties_stack.setCurrentIndex(3)
            current_mode_msg = self.statusBar().currentMessage().split(":")[0] if ":" in self.statusBar().currentMessage() else "Ready"
            self.statusBar().showMessage(f"{current_mode_msg}")
        elif len(selected_neuron_names_set) == 1:
            self.selected_item = list(selected_neuron_names_set)[0]
            self.properties_stack.setCurrentIndex(0)
            self.update_property_panel()
            self.statusBar().showMessage(f"Selected: {self.selected_item}")
        else:
            self.selected_item = None
            self.properties_stack.setCurrentIndex(2)
            self.statusBar().showMessage(f"Selected {len(selected_neuron_names_set)} neurons.")

    def get_connection_at_pos(self, logical_pos: QtCore.QPointF, threshold=10.0):
        scaled_threshold = threshold
        for (source_name, target_name), conn_obj in self.network.connections.items():
            if source_name in self.network.neurons and target_name in self.network.neurons:
                p1_logical = QtCore.QPointF(*self.network.neurons[source_name].get_position())
                p2_logical = QtCore.QPointF(*self.network.neurons[target_name].get_position())
                if self.is_point_near_line_segment(logical_pos, p1_logical, p2_logical, scaled_threshold):
                    return (source_name, target_name)
        return None

    def is_point_near_line_segment(self, p: QtCore.QPointF, a: QtCore.QPointF, b: QtCore.QPointF, threshold_dist: float):
        ab_x = b.x() - a.x(); ab_y = b.y() - a.y()
        ap_x = p.x() - a.x(); ap_y = p.y() - a.y()
        ab_len_sq = ab_x * ab_x + ab_y * ab_y
        if ab_len_sq == 0: return math.hypot(ap_x, ap_y) <= threshold_dist
        t = (ap_x * ab_x + ap_y * ab_y) / ab_len_sq
        if t < 0: closest_point_on_line = a
        elif t > 1: closest_point_on_line = b
        else: closest_point_on_line = QtCore.QPointF(a.x() + t * ab_x, a.y() + t * ab_y)
        return math.hypot(p.x() - closest_point_on_line.x(), p.y() - closest_point_on_line.y()) <= threshold_dist

    def visualization_mouse_press(self, event: QtGui.QMouseEvent):
        logical_pos = self.vis._widget_to_logical(event.pos())
        neuron_under_cursor = self.vis.get_neuron_at_pos(event.pos())
        conn_under_cursor = self.get_connection_at_pos(logical_pos) if not neuron_under_cursor else None
        
        if event.button() == QtCore.Qt.LeftButton:
            if self.mode == "add_neuron": 
                self.add_neuron_at_position(logical_pos.x(), logical_pos.y())
            elif self.mode == "add_connection" and neuron_under_cursor:
                self.connection_start_neuron = neuron_under_cursor
                self.statusBar().showMessage(f"Connection: Start from '{neuron_under_cursor}'. Drag to target neuron.")
            elif self.mode == "remove":
                if neuron_under_cursor: self.handle_remove_neuron_action(neuron_under_cursor)
                elif conn_under_cursor: self.remove_connection(*conn_under_cursor)
            elif self.mode == "select":
                # FIX: Handle starting a drag operation
                if neuron_under_cursor:
                    self.vis.dragged_neuron = neuron_under_cursor
                    neuron_pos = QtCore.QPointF(*self.network.neurons[neuron_under_cursor].get_position())
                    self.vis.drag_offset = logical_pos - neuron_pos
                    self.on_neuron_vis_clicked_for_property_panel(neuron_under_cursor) # also select it
                elif conn_under_cursor:
                    self.vis.selected_neurons.clear()
                    self.selected_item = conn_under_cursor
                    self.update_property_panel()
                    self.vis.selectionChanged.emit(set())
                    self.vis.update()
                else:
                    self.clear_selection_action()


    def visualization_mouse_move(self, event: QtGui.QMouseEvent):
        # FIX: Handle the drag movement
        if self.mode == "select" and self.vis.dragged_neuron:
            logical_pos = self.vis._widget_to_logical(event.pos())
            neuron = self.network.neurons.get(self.vis.dragged_neuron)
            if neuron:
                new_pos = logical_pos - self.vis.drag_offset
                neuron.set_position(new_pos.x(), new_pos.y())
                self.update_property_panel()
                self.vis.update()

    def visualization_mouse_release(self, event: QtGui.QMouseEvent):
        # FIX: Finalize the drag operation
        if event.button() == QtCore.Qt.LeftButton:
            if self.mode == "select":
                self.vis.dragged_neuron = None
            elif self.mode == "add_connection" and self.connection_start_neuron:
                target_neuron_name = self.vis.get_neuron_at_pos(event.pos())
                if target_neuron_name and target_neuron_name != self.connection_start_neuron:
                    success = self.network.connect(self.connection_start_neuron, target_neuron_name, random.uniform(0.1, 0.5))
                    if success:
                         self.statusBar().showMessage(f"Connected: '{self.connection_start_neuron}' -> '{target_neuron_name}'")
                         self.update_network_statistics()
                    else: self.statusBar().showMessage(f"Failed to connect.")
                else: self.statusBar().showMessage("Connection cancelled.")
                self.connection_start_neuron = None
                self.vis.update()

    def update_property_panel(self):
        item = self.selected_item
        if isinstance(item, str) and item in self.network.neurons:
            self.properties_stack.setCurrentIndex(0)
            n_obj = self.network.neurons[item]; pos_tup = n_obj.get_position()
            widgets = [self.neuron_name_edit, self.neuron_value_spin, self.neuron_type_props_combo, self.neuron_x_spin, self.neuron_y_spin]
            for w in widgets: w.blockSignals(True)
            self.neuron_name_edit.setText(n_obj.name); self.neuron_value_spin.setValue(self.network.state.get(item, 50.0))
            self.neuron_type_props_combo.setCurrentText(n_obj.type); self.neuron_x_spin.setValue(int(pos_tup[0])); self.neuron_y_spin.setValue(int(pos_tup[1]))
            for w in widgets: w.blockSignals(False)
        elif isinstance(item, tuple) and len(item) == 2 and item in self.network.connections:
            self.properties_stack.setCurrentIndex(1)
            src_n, tgt_n = item; conn_obj = self.network.connections[item]
            self.connection_source_label.setText(src_n); self.connection_target_label.setText(tgt_n)
            self.connection_weight_spin.blockSignals(True); self.connection_weight_spin.setValue(conn_obj.get_weight()); self.connection_weight_spin.blockSignals(False)
        elif self.vis.selected_neurons and len(self.vis.selected_neurons) > 1: self.properties_stack.setCurrentIndex(2)
        else: self.properties_stack.setCurrentIndex(3)

    def update_neuron_property_from_panel(self):
        if not (isinstance(self.selected_item, str) and self.selected_item in self.network.neurons): return
        original_name = self.selected_item; neuron = self.network.neurons.get(original_name)
        if not neuron: return
        new_name = self.neuron_name_edit.text().strip()
        new_type = self.neuron_type_props_combo.currentText()
        new_val = float(self.neuron_value_spin.value())
        new_pos = (self.neuron_x_spin.value(), self.neuron_y_spin.value())
        changed = False; renamed_inspector_target = None
        if neuron.get_position() != new_pos: neuron.set_position(new_pos[0], new_pos[1]); self.vis._update_layer_rects(); changed = True
        if neuron.type != new_type:
            neuron.type = new_type
            cfg_app = self.network.config.neurogenesis['appearance']
            neuron.attributes['shape'] = cfg_app['shapes'].get(new_type, cfg_app['shapes']['default'])
            neuron.attributes['color'] = cfg_app['colors'].get(new_type, cfg_app['colors']['default'])
            changed = True
        if self.network.state.get(original_name) != new_val: self.network.state[original_name] = new_val; changed = True
        if new_name and new_name != original_name:
            if new_name not in self.network.neurons:
                self.network.neurons[new_name] = self.network.neurons.pop(original_name); neuron.name = new_name
                if original_name in self.network.state: self.network.state[new_name] = self.network.state.pop(original_name)
                conns_to_remap = {}
                for (s, t), conn_obj_ref in list(self.network.connections.items()):
                    ns, nt = s, t
                    if s == original_name: ns = new_name
                    if t == original_name: nt = new_name
                    if ns != s or nt != t: conn_obj_ref.source, conn_obj_ref.target = ns, nt; conns_to_remap[(s,t)] = (ns,nt)
                for old_k, new_k in conns_to_remap.items(): self.network.connections[new_k] = self.network.connections.pop(old_k)
                for l_data in self.layers.values():
                    if original_name in l_data['neurons']: l_data['neurons'] = [new_name if n == original_name else n for n in l_data['neurons']]
                self.selected_item = new_name; renamed_inspector_target = new_name; changed = True
                if original_name in self.vis.selected_neurons: self.vis.selected_neurons.remove(original_name); self.vis.selected_neurons.add(new_name)
            else:
                self.neuron_name_edit.setText(original_name)
                QtWidgets.QMessageBox.warning(self, "Rename Error", f"Name '{new_name}' is invalid or already exists.")
        if changed:
            if renamed_inspector_target: self.update_simulation_combo()
            current_inspector_target = renamed_inspector_target or original_name
            if current_inspector_target in self.active_inspectors: self.active_inspectors[current_inspector_target].update_neuron_reference(current_inspector_target)
            self.vis.update()

    def update_connection_property_from_panel(self):
        if isinstance(self.selected_item, tuple) and len(self.selected_item) == 2:
            conn_key = self.selected_item
            if conn_key in self.network.connections:
                new_weight = float(self.connection_weight_spin.value())
                self.network.connections[conn_key].set_weight(new_weight)
                src, tgt = conn_key
                if src in self.active_inspectors: self.active_inspectors[src].populate_connections_tab()
                if tgt in self.active_inspectors: self.active_inspectors[tgt].populate_connections_tab()
                self.vis.update()

    def add_neuron_at_position(self, x_logical, y_logical):
        default_name = f"N_{self.neuron_counter}"
        neuron_name_str, ok = QtWidgets.QInputDialog.getText(self, "Add Neuron", "Neuron Name:", text=default_name)
        if not ok or not neuron_name_str.strip(): self.statusBar().showMessage("Neuron addition cancelled."); return
        neuron_name_str = neuron_name_str.strip()
        if neuron_name_str in self.network.neurons: QtWidgets.QMessageBox.warning(self, "Add Error", f"Name '{neuron_name_str}' exists."); return
        self.neuron_counter += 1
        n_type = self.neuron_type_combo_adder.currentText()
        cfg_app = self.network.config.neurogenesis['appearance']
        color = cfg_app['colors'].get(n_type, cfg_app['colors']['default'])
        shape = cfg_app['shapes'].get(n_type, cfg_app['shapes']['default'])
        self.network.add_neuron(neuron_name_str, 50.0, (x_logical, y_logical), n_type, {'shape': shape, 'color': color})
        self.update_simulation_combo(); self.vis.highlight_new_neuron(neuron_name_str, 2.0); self.update_network_statistics(); self.vis.update()
        self.statusBar().showMessage(f"Added: '{neuron_name_str}' at ({x_logical:.0f}, {y_logical:.0f})")

    def open_neuron_inspector(self, neuron_name):
        if neuron_name in self.active_inspectors and self.active_inspectors[neuron_name].isVisible():
            self.active_inspectors[neuron_name].raise_(); self.active_inspectors[neuron_name].activateWindow()
            return
        if neuron_name in self.network.neurons:
            dialog = NeuronInspectorDialog(neuron_name, self.network, self)
            dialog.neuronPropertyChanged.connect(self.handle_inspector_property_change)
            dialog.finished.connect(lambda name=neuron_name: self.active_inspectors.pop(name, None))
            self.active_inspectors[neuron_name] = dialog; dialog.show()

    def handle_inspector_property_change(self, original_neuron_name, prop_name, new_value):
        target_neuron_name = original_neuron_name; neuron = self.network.neurons.get(original_neuron_name)
        if not neuron and prop_name!="name": print(f"Inspector change for non-existent '{original_neuron_name}'."); return
        print(f"GUI inspector change: N={original_neuron_name}, P={prop_name}, V={new_value}"); renamed_to=None
        if prop_name=="name":
            new_name_str=str(new_value).strip()
            if new_name_str and new_name_str!=original_neuron_name and new_name_str not in self.network.neurons:
                neuron_obj_to_rename=self.network.neurons.pop(original_neuron_name)
                neuron_obj_to_rename.name=new_name_str; self.network.neurons[new_name_str]=neuron_obj_to_rename
                target_neuron_name=new_name_str; renamed_to=new_name_str
                if original_neuron_name in self.network.state:self.network.state[new_name_str]=self.network.state.pop(original_neuron_name)
                conns_to_remap={}
                for(s,t),conn_obj in list(self.network.connections.items()):
                    ns,nt=s,t
                    if s==original_neuron_name:ns=new_name_str
                    if t==original_neuron_name:nt=new_name_str
                    if ns!=s or nt!=t:conn_obj.source,conn_obj.target=ns,nt;conns_to_remap[(s,t)]=(ns,nt)
                for old_key,new_key in conns_to_remap.items():self.network.connections[new_key]=self.network.connections.pop(old_key)
                for l_data in self.layers.values():
                    if original_neuron_name in l_data['neurons']:l_data['neurons']=[new_name_str if n==original_neuron_name else n for n in l_data['neurons']]
                if self.selected_item==original_neuron_name:self.selected_item=new_name_str
                if original_neuron_name in self.vis.selected_neurons:self.vis.selected_neurons.remove(original_neuron_name);self.vis.selected_neurons.add(new_name_str)
                if original_neuron_name in self.active_inspectors:
                    inspector=self.active_inspectors.pop(original_neuron_name)
                    inspector.update_neuron_reference(new_name_str); self.active_inspectors[new_name_str]=inspector
                self.statusBar().showMessage(f"Neuron '{original_neuron_name}' renamed to '{new_name_str}'")
            elif new_name_str!=original_neuron_name:
                if original_neuron_name in self.active_inspectors:self.active_inspectors[original_neuron_name].name_edit_inspector.setText(original_neuron_name)
                QtWidgets.QMessageBox.warning(self,"Rename Fail",f"Cannot rename to '{new_name_str}'.");return
        current_neuron_obj=self.network.neurons.get(target_neuron_name)
        if not current_neuron_obj:return
        if prop_name=="type":current_neuron_obj.type=str(new_value)
        elif prop_name=="state_value":self.network.state[target_neuron_name]=float(new_value)
        elif prop_name=="position":
            if isinstance(new_value,(tuple,list))and len(new_value)==2:current_neuron_obj.set_position(float(new_value[0]),float(new_value[1]));self.vis._update_layer_rects()
        elif prop_name=="attribute_color":
            if isinstance(new_value,(tuple,list))and len(new_value)==3:current_neuron_obj.attributes['color']=tuple(map(int,new_value))
        elif prop_name=="attribute_shape":current_neuron_obj.attributes['shape']=str(new_value)
        elif prop_name=="connection_weight":
            conn_key,weight_val=new_value
            if conn_key in self.network.connections:self.network.connections[conn_key].set_weight(float(weight_val))
        if self.selected_item==original_neuron_name or self.selected_item==target_neuron_name:self.selected_item=target_neuron_name;self.update_property_panel()
        if renamed_to:self.update_simulation_combo()
        self.update_network_statistics();self.vis.update()

    def handle_remove_neuron_action(self, neuron_name):
        if neuron_name in self.network.neurons: self.remove_neuron_logic(neuron_name)

    def remove_neuron_logic(self, name_to_remove):
        if name_to_remove in self.active_inspectors: self.active_inspectors.pop(name_to_remove).close()
        self.network.connections={k:v for k,v in self.network.connections.items() if name_to_remove not in k}
        if name_to_remove in self.network.neurons: del self.network.neurons[name_to_remove]
        if name_to_remove in self.network.state: del self.network.state[name_to_remove]
        for l_data in self.layers.values():
            if name_to_remove in l_data['neurons']: l_data['neurons'].remove(name_to_remove)
        self.update_simulation_combo()
        if self.selected_item==name_to_remove:self.clear_selection_action()
        if name_to_remove in self.vis.selected_neurons:self.vis.selected_neurons.remove(name_to_remove);self.vis.selectionChanged.emit(self.vis.selected_neurons)
        self.update_network_statistics();self.vis._update_layer_rects();self.vis.update()
        self.statusBar().showMessage(f"Removed Neuron: '{name_to_remove}'")

    def remove_connection(self, source_name, target_name):
        conn_key=(source_name, target_name)
        if conn_key in self.network.connections:
            del self.network.connections[conn_key]
            if self.selected_item==conn_key:self.clear_selection_action()
            self.update_network_statistics();self.vis.update()
            self.statusBar().showMessage(f"Removed Connection: '{source_name}' -> '{target_name}'")

    def select_color_dialog(self, button_to_update_style, color_list_reference):
        current_qcolor=QtGui.QColor(*color_list_reference)
        chosen_color=QtWidgets.QColorDialog.getColor(current_qcolor, self, "Select Layer Color")
        if chosen_color.isValid():
            color_list_reference[0:3]=[chosen_color.red(),chosen_color.green(),chosen_color.blue()]
            button_to_update_style.setStyleSheet(f"background-color: {chosen_color.name()};")
            button_to_update_style.setText(chosen_color.name())

    def add_layer_dialog(self):
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle("Add Layer");layout=QtWidgets.QFormLayout(dialog)
        prefix_edit=QtWidgets.QLineEdit(f"layer{self.layer_counter}_")
        count_spin=QtWidgets.QSpinBox();count_spin.setRange(1,100);count_spin.setValue(3);count_spin.setToolTip("Num neurons.")
        x_spin=QtWidgets.QSpinBox();x_spin.setRange(-10000,10000);x_spin.setValue(100+self.layer_counter*300);x_spin.setToolTip("X pos.")
        y_spin=QtWidgets.QSpinBox();y_spin.setRange(-10000,10000);y_spin.setValue(150);y_spin.setToolTip("Y start.")
        space_spin=QtWidgets.QSpinBox();space_spin.setRange(30,500);space_spin.setValue(100);space_spin.setToolTip("V spacing.")
        type_combo=QtWidgets.QComboBox();type_combo.addItems(["default","input","hidden","output"]);type_combo.setToolTip("Neuron type.")
        color_list=[random.randint(180,230) for _ in range(3)]
        color_btn=QtWidgets.QPushButton(f"rgb({color_list[0]},{color_list[1]},{color_list[2]})")
        color_btn.setStyleSheet(f"background-color: rgb({color_list[0]},{color_list[1]},{color_list[2]});")
        color_btn.clicked.connect(lambda: self.select_color_dialog(color_btn, color_list))
        layout.addRow("Prefix:", prefix_edit); layout.addRow("Count:", count_spin); layout.addRow("X:",x_spin); layout.addRow("Y:",y_spin)
        layout.addRow("Spacing:",space_spin); layout.addRow("Type:",type_combo); layout.addRow("Color:",color_btn)
        buttons=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel);buttons.accepted.connect(dialog.accept);buttons.rejected.connect(dialog.reject);layout.addRow(buttons)
        if dialog.exec_():
            p,c,x,y,s,t=prefix_edit.text(),count_spin.value(),x_spin.value(),y_spin.value(),space_spin.value(),type_combo.currentText()
            ln, lname = [], f"layer{self.layer_counter}"
            fc, fqc = tuple(color_list), QtGui.QColor(*color_list)
            cfg_s = self.network.config.neurogenesis['appearance']['shapes']
            def_s = cfg_s.get(t, cfg_s['default'])
            for i in range(c):
                n_base = f"{p}{i}"; actual_n = n_base; idx=0
                while actual_n in self.network.neurons: actual_n = f"{n_base}_{idx}"; idx+=1
                yp = y+i*s; attrs={'shape':def_s, 'color':fc, 'layer':lname}
                self.network.add_neuron(actual_n,50.0,(x,yp),t,attrs); ln.append(actual_n)
            self.layers[lname]={'neurons':ln,'color':fqc.name()}; self.layer_counter+=1
            self.update_simulation_combo(); self.vis.set_layers_data(self.layers); self.update_network_statistics()
            self.statusBar().showMessage(f"Added layer '{lname}'")

    def connect_layers_dialog(self):
        if len(self.layers)<2: QtWidgets.QMessageBox.warning(self,"Connect Error","Need >= 2 layers.");return
        dialog=QtWidgets.QDialog(self);dialog.setWindowTitle("Connect Layers");layout=QtWidgets.QFormLayout(dialog)
        src_c,tgt_c=QtWidgets.QComboBox(),QtWidgets.QComboBox(); keys=list(self.layers.keys())
        src_c.addItems(keys);tgt_c.addItems(keys)
        if len(keys)>=2:src_c.setCurrentIndex(0);tgt_c.setCurrentIndex(1)
        type_c=QtWidgets.QComboBox();type_c.addItems(["Fully Connected","One-to-One (Matching Size)"])
        min_w,max_w=QtWidgets.QDoubleSpinBox(),QtWidgets.QDoubleSpinBox()
        for s_w,v in [(min_w,-0.5),(max_w,0.5)]:s_w.setRange(-1,1);s_w.setValue(v);s_w.setSingleStep(0.01);s_w.setDecimals(3)
        layout.addRow("Src:",src_c);layout.addRow("Tgt:",tgt_c);layout.addRow("Type:",type_c)
        layout.addRow("MinW:",min_w);layout.addRow("MaxW:",max_w)
        btns=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel);btns.accepted.connect(dialog.accept);btns.rejected.connect(dialog.reject);layout.addRow(btns)
        if dialog.exec_():
            sname,tname=src_c.currentText(),tgt_c.currentText()
            if sname==tname:QtWidgets.QMessageBox.warning(self,"Error","Layers must differ.");return
            s_n,t_n=self.layers[sname]['neurons'],self.layers[tname]['neurons']
            ctype,w_min_v,w_max_v=type_c.currentText(),min_w.value(),max_w.value();added_c=0
            if ctype=="Fully Connected":
                for s_neuron in s_n:
                    for t_neuron in t_n: 
                        if self.network.connect(s_neuron,t_neuron,random.uniform(w_min_v,w_max_v)):added_c+=1
            elif ctype=="One-to-One (Matching Size)":
                if len(s_n)!=len(t_n):QtWidgets.QMessageBox.warning(self,"Error","One-to-One size mismatch.");return
                for s_neuron,t_neuron in zip(s_n,t_n):
                    if self.network.connect(s_neuron,t_neuron,random.uniform(w_min_v,w_max_v)):added_c+=1
            self.update_network_statistics();self.vis.update();self.statusBar().showMessage(f"Added {added_c} conns.")

    def create_feedforward_dialog(self):
        text,ok=QtWidgets.QInputDialog.getText(self,"Create Feedforward Network","Layer sizes (comma-separated, e.g., 2,3,1 for Input, Hidden, Output):",QtWidgets.QLineEdit.Normal, "2,3,1")
        if ok and text:
            try:
                sizes=[int(s.strip()) for s in text.split(',') if s.strip().isdigit() and int(s.strip())>0]
                if len(sizes)<2:raise ValueError("Need input & output layers.")
                reply=QtWidgets.QMessageBox.question(self,"Confirm","Clear current net first?",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No|QtWidgets.QMessageBox.Cancel)
                if reply==QtWidgets.QMessageBox.Cancel:return
                if reply==QtWidgets.QMessageBox.Yes:self.clear_network_action(confirm=False)
                vis_h_log=self.vis.height()/self.vis.zoom_factor;y_center=vis_h_log/2-self.vis.pan_offset.y()
                self.create_feedforward_network_structure(sizes,100,250,y_center,-0.7,0.7)
                self.statusBar().showMessage(f"Created feedforward net: {sizes}")
            except ValueError as e:QtWidgets.QMessageBox.warning(self,"Input Error",str(e))

    def create_feedforward_network_structure(self, layer_sizes, x_start, x_space, y_center, min_w, max_w):
        self.layers={};all_layers_neurons=[]
        types=["input"]+["hidden"]*(len(layer_sizes)-2)+["output"]
        neuron_v_spacing=100
        for idx,(size,n_type) in enumerate(zip(layer_sizes,types)):
            current_neurons,name=[],f"ff_layer{idx}_{n_type}"
            x_pos=x_start+idx*x_space
            total_h=(size-1)*neuron_v_spacing;y_s=y_center-total_h/2.0
            cfg_app=self.network.config.neurogenesis['appearance']
            color=cfg_app['colors'].get(n_type,cfg_app['colors']['default'])
            shape=cfg_app['shapes'].get(n_type,cfg_app['shapes']['default'])
            for i in range(size):
                n_name=f"{n_type}{idx}_{i}";y_pos=y_s+i*neuron_v_spacing
                attrs={'shape':shape,'color':color,'layer':name}
                self.network.add_neuron(n_name,50.0,(x_pos,y_pos),n_type,attrs);current_neurons.append(n_name)
            self.layers[name]={'neurons':current_neurons,'color':QtGui.QColor(*color).name()}
            all_layers_neurons.append(current_neurons)
        for i in range(len(all_layers_neurons)-1):
            for src_n in all_layers_neurons[i]:
                for tgt_n in all_layers_neurons[i+1]:self.network.connect(src_n,tgt_n,random.uniform(min_w,max_w))
        self.neuron_counter=len(self.network.neurons);self.layer_counter=len(self.layers)
        self.update_simulation_combo();self.vis.set_layers_data(self.layers);self.update_network_statistics();self.vis.update()

    def auto_layout_network(self):
        if not self.network.neurons:self.statusBar().showMessage("No neurons.");return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.statusBar().showMessage("Auto-layout...");QtWidgets.QApplication.processEvents()
        vis_w,vis_h=self.vis.width()/self.vis.zoom_factor,self.vis.height()/self.vis.zoom_factor
        area=vis_w*vis_h;k=math.sqrt(area/max(1,len(self.network.neurons)))
        iters,temp=100,vis_w/10.0;cool_f=0.97
        pos={n:QtCore.QPointF(*neuron.get_position()) for n,neuron in self.network.neurons.items()}
        names=list(self.network.neurons.keys())
        for i in range(iters):
            disp={n:QtCore.QPointF(0,0) for n in names}
            for idx1 in range(len(names)):
                n1=names[idx1]
                for idx2 in range(idx1+1,len(names)):
                    n2=names[idx2];delta=pos[n1]-pos[n2];dist=math.hypot(delta.x(),delta.y())or 0.01
                    force=(k*k)/dist;disp[n1]+=(delta/dist)*force;disp[n2]-=(delta/dist)*force
            for (s,t),_ in self.network.connections.items():
                if s in pos and t in pos:
                    delta=pos[t]-pos[s];dist=math.hypot(delta.x(),delta.y())or 0.01
                    force=(dist*dist)/k;disp[s]+=(delta/dist)*force;disp[t]-=(delta/dist)*force
            for n in names:
                d_vec=disp[n];d_mag=math.hypot(d_vec.x(),d_vec.y())or 0.01
                new_p=pos[n]+(d_vec/d_mag)*min(d_mag,temp)
                cx_log,cy_log=-self.vis.pan_offset.x()+vis_w/1.6,-self.vis.pan_offset.y()+vis_h/1.6
                new_p.setX(max(cx_log-vis_w/2+50,min(cx_log+vis_w/2-50,new_p.x())))
                new_p.setY(max(cy_log-vis_h/2+50,min(cy_log+vis_h/2-50,new_p.y())))
                pos[n]=new_p
            temp*=cool_f
            if i%10==0:
                for n_name,p_qf in pos.items():self.network.neurons[n_name].set_position(p_qf.x(),p_qf.y())
                self.vis._update_layer_rects();self.vis.update();QtWidgets.QApplication.processEvents()
        for n_name,p_qf in pos.items():self.network.neurons[n_name].set_position(p_qf.x(),p_qf.y())
        self.vis._update_layer_rects();self.vis.update();QtWidgets.QApplication.restoreOverrideCursor()
        self.statusBar().showMessage("Auto-layout applied.")

    def update_single_neuron_state_from_controls(self):
        name=self.update_neuron_combo.currentText()
        if name!="Select Neuron" and name in self.network.neurons:
            val=float(self.update_value_spin.value())
            self.network.state[name]=val
            self.vis.highlight_neuron(name,1.5,'activity');self.vis.update()
            self.statusBar().showMessage(f"Set '{name}' activation to {val:.2f}.")
            if name in self.active_inspectors:self.active_inspectors[name].populate_all_data()
        else:self.statusBar().showMessage("Select a valid neuron.")

    def perform_learning_action(self):
        updated=self.network.perform_learning()
        if updated is not None:
            msg=f"Hebbian: {len(updated)} pairs updated." if updated else "Hebbian: No co-activity."
            for n1,n2 in (updated or []):
                for name in [n1,n2]:
                    if name in self.active_inspectors:self.active_inspectors[name].populate_connections_tab()
        else:msg="Hebbian: Skipped (too soon)."
        self.statusBar().showMessage(msg);self.vis.update()

    def propagate_activation_action(self):
        self.network.propagate_activation();self.statusBar().showMessage("Activation propagated.")
        for name in self.network.neurons:self.vis.highlight_neuron(name,0.5,'activity_pulse')
        for insp in self.active_inspectors.values():insp.populate_all_data()
        self.vis.update()

    def new_network_action(self):
        if QtWidgets.QMessageBox.question(self,"New","Clear current network?",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)==QtWidgets.QMessageBox.Yes:
            self.clear_network_action(confirm=False)

    def open_network_action(self):
        path,_=QtWidgets.QFileDialog.getOpenFileName(self,"Open Network","","JSON (*.json)")
        if path:
            net=Network.load(path)
            if net:
                self.clear_network_action(confirm=False)
                self.network=net;self.vis.network=net;self.layers={}
                max_idx=-1
                for n_name,n_obj in self.network.neurons.items():
                    l_attr=n_obj.attributes.get('layer')
                    if l_attr:
                        if l_attr not in self.layers:
                            color=n_obj.attributes.get('color',(200,200,200))
                            self.layers[l_attr]={'neurons':[],'color':QtGui.QColor(*color).name()}
                            if l_attr.startswith("layer") and l_attr[5:].isdigit(): max_idx=max(max_idx,int(l_attr[5:]))
                        self.layers[l_attr]['neurons'].append(n_name)
                self.layer_counter=max_idx+1;self.neuron_counter=len(self.network.neurons)
                self.update_simulation_combo();self.vis.set_layers_data(self.layers);self.update_network_statistics();self.clear_selection_action()
                self.lr_spin.setValue(self.network.config.hebbian.get('base_learning_rate',0.1))
                self.active_thresh_spin.setValue(self.network.config.hebbian.get('active_threshold',50))
                self.hebbian_interval_spin.setValue(self.network.config.hebbian.get('learning_interval',30000))
                self.neurogenesis_enable_cb.setChecked(self.network.neurogenesis_enabled) 
                self.vis.update();self.statusBar().showMessage(f"Loaded: {os.path.basename(path)}")
            else:QtWidgets.QMessageBox.warning(self,"Error",f"Failed to load {path}")

    def save_network_action(self):
        default_fn="neural_network.json"
        path,_=QtWidgets.QFileDialog.getSaveFileName(self,"Save Network",default_fn,"JSON (*.json)")
        if path:
            if not path.endswith(".json"):path+=".json"
            self.network.config.neurogenesis['enabled_globally'] = self.network.neurogenesis_enabled
            if self.network.save(path):self.statusBar().showMessage(f"Saved: {os.path.basename(path)}")
            else:QtWidgets.QMessageBox.warning(self,"Error",f"Failed to save to {path}")

    def clear_network_action(self,confirm=True):
        if confirm:
            if QtWidgets.QMessageBox.question(self,"Clear","Clear entire network?",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)==QtWidgets.QMessageBox.No:return
        for insp in list(self.active_inspectors.values()):insp.close()
        self.active_inspectors.clear()
        self.network=Network();self.vis.network=self.network;self.layers={}
        self.neuron_counter=0;self.layer_counter=0
        self.update_simulation_combo();self.clear_selection_action();self.vis.set_layers_data(self.layers);self.update_network_statistics()
        self.lr_spin.setValue(self.network.config.hebbian.get('base_learning_rate',0.1))
        self.active_thresh_spin.setValue(self.network.config.hebbian.get('active_threshold',50))
        self.hebbian_interval_spin.setValue(self.network.config.hebbian.get('learning_interval',30000))
        self.neurogenesis_enable_cb.setChecked(self.network.neurogenesis_enabled) 
        self.vis.update();self.statusBar().showMessage("Network Cleared.")

    def randomize_weights_action(self):
        if not self.network.connections:self.statusBar().showMessage("No connections.");return
        for conn in self.network.connections.values():conn.set_weight(random.uniform(-1.0,1.0))
        for insp in self.active_inspectors.values():insp.populate_connections_tab()
        self.vis.update();self.statusBar().showMessage("Weights randomized.")

    def show_about_dialog(self):
        QtWidgets.QMessageBox.about(self,"About","NN Builder v1.1\nVisual Neural Network Editor.")

    def update_network_statistics(self):
        self.stats_label.setText(f"Neurons: {len(self.network.neurons)} | Connections: {len(self.network.connections)}")

    def update_simulation_combo(self):
        current=self.update_neuron_combo.currentText();self.update_neuron_combo.blockSignals(True)
        self.update_neuron_combo.clear();self.update_neuron_combo.addItem("Select Neuron")
        names=sorted(self.network.neurons.keys())
        if names:self.update_neuron_combo.addItems(names)
        if current in names:self.update_neuron_combo.setCurrentText(current)
        elif names:self.update_neuron_combo.setCurrentIndex(1)
        else:self.update_neuron_combo.setCurrentIndex(0)
        self.update_neuron_combo.blockSignals(False)

    def closeEvent(self,event:QtGui.QCloseEvent):
        for insp in list(self.active_inspectors.values()):insp.close()
        if QtWidgets.QMessageBox.question(self,'Exit',"Sure to exit?",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)==QtWidgets.QMessageBox.Yes:event.accept()
        else:event.ignore()

def main():
    app = QtWidgets.QApplication(sys.argv)
    if "Fusion" in QtWidgets.QStyleFactory.keys(): app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    window = NetworkBuilderGUI()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path: sys.path.insert(0, current_dir)
    main()