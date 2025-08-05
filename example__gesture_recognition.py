# example__gesture_recognition.py
import sys
import os
import numpy as np
import random
import cv2
from PyQt5 import QtWidgets, QtCore, QtGui
from NeuralNetwork.core import Network
from NeuralNetwork.learning import BackpropNetwork

class GestureRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.gestures = ["fist", "peace", "thumbs_up", "open_hand"]
        self.setup_network()
        self.setup_ui()
        self.detect_available_cameras()
        self.setup_webcam()
        self.start_processing()
        
    def setup_ui(self):
        self.setWindowTitle("Gesture Recognition System")
        self.setGeometry(100, 100, 1200, 700)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)
        
        # Splitter for resizable panes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # Webcam and ROI display
        self.webcam_label = QtWidgets.QLabel("Initializing Webcam...")
        self.webcam_label.setMinimumSize(640, 480)
        self.roi_label = QtWidgets.QLabel()
        self.roi_label.setMinimumSize(128, 128)
        
        # Control panel
        control_panel = QtWidgets.QWidget()
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        
        # Camera selection controls
        camera_group = QtWidgets.QGroupBox("Camera Settings")
        camera_layout = QtWidgets.QFormLayout(camera_group)
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.switch_camera)
        camera_layout.addRow("Select Camera:", self.camera_combo)
        
        # Gesture training controls
        training_group = QtWidgets.QGroupBox("Gesture Training")
        training_layout = QtWidgets.QFormLayout(training_group)
        self.gesture_combo = QtWidgets.QComboBox()
        self.gesture_combo.addItems([g.capitalize() for g in self.gestures])
        training_layout.addRow("Gesture:", self.gesture_combo)
        self.capture_btn = QtWidgets.QPushButton("Capture Sample")
        self.capture_btn.clicked.connect(self.capture_gesture_sample)
        training_layout.addRow(self.capture_btn)
        self.sample_count_label = QtWidgets.QLabel("Samples: 0")
        training_layout.addRow(self.sample_count_label)
        self.train_btn = QtWidgets.QPushButton("Train Network")
        self.train_btn.clicked.connect(self.train_network)
        training_layout.addRow(self.train_btn)
        
        # Recognition results
        result_group = QtWidgets.QGroupBox("Recognition Results")
        result_layout = QtWidgets.QVBoxLayout(result_group)
        self.gesture_label = QtWidgets.QLabel("No gesture detected")
        self.gesture_label.setStyleSheet("font-size: 24px; font-weight: bold;")
        self.gesture_label.setAlignment(QtCore.Qt.AlignCenter)
        result_layout.addWidget(self.gesture_label)
        
        # Activation visualization
        self.activation_bars = {}
        activation_layout = QtWidgets.QGridLayout()
        for i, gesture in enumerate(self.gestures):
            label = QtWidgets.QLabel(gesture.capitalize())
            progress = QtWidgets.QProgressBar()
            progress.setRange(0, 100)
            progress.setTextVisible(False)
            self.activation_bars[gesture] = progress
            activation_layout.addWidget(label, i, 0)
            activation_layout.addWidget(progress, i, 1)
        result_layout.addLayout(activation_layout)
        
        control_layout.addWidget(camera_group)
        control_layout.addWidget(training_group)
        control_layout.addWidget(result_group)
        control_layout.addStretch()
        
        # Layout setup
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.addWidget(self.webcam_label)
        left_layout.addWidget(self.roi_label)
        
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(control_panel)
        self.splitter.setSizes([700, 500])
        main_layout.addWidget(self.splitter)
        
        self.statusBar().showMessage("Ready")
        
    def setup_network(self):
        self.network = Network()
        self.network.config.hebbian['learning_interval'] = float('inf')
        self.network.set_neurogenesis_enabled(False)
        
        # Input layer (16x16 grayscale image = 256 neurons)
        self.input_neurons = [f"pixel_{i}" for i in range(256)]
        for i, name in enumerate(self.input_neurons):
            x = 50 + (i % 16) * 20
            y = 50 + (i // 16) * 20
            self.network.add_neuron(name, 0, (x, y), "input")
        
        # Hidden layer
        self.hidden_neurons = [f"hidden_{i}" for i in range(32)]
        for i, name in enumerate(self.hidden_neurons):
            self.network.add_neuron(name, 0, (400, 50 + i * 30), "hidden")
        
        # Output layer (one neuron per gesture)
        self.output_neurons = [f"{g}_out" for g in self.gestures]
        for i, name in enumerate(self.output_neurons):
            self.network.add_neuron(name, 0, (600, 100 + i * 60), "output")
        
        # Connections
        for input_n in self.input_neurons:
            for hidden_n in self.hidden_neurons:
                self.network.connect(input_n, hidden_n, random.uniform(-0.2, 0.2))
                
        for hidden_n in self.hidden_neurons:
            for output_n in self.output_neurons:
                self.network.connect(hidden_n, output_n, random.uniform(-0.2, 0.2))
        
        self.backprop_learner = BackpropNetwork(self.network, learning_rate=0.1)
        self.backprop_learner.set_layers([self.input_neurons, self.hidden_neurons, self.output_neurons])
        
        self.training_data = []
        
    def detect_available_cameras(self):
        self.available_cameras = []
        # Try to detect up to 5 cameras
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.available_cameras.append(i)
                cap.release()
        
        if not self.available_cameras:
            QtWidgets.QMessageBox.critical(self, "Error", "No cameras detected!")
            return
            
        self.current_camera_index = self.available_cameras[0]
    
    def setup_webcam(self):
        if not hasattr(self, 'current_camera_index'):
            self.current_camera_index = 0
            
        self.cap = cv2.VideoCapture(self.current_camera_index)
        if not self.cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not access camera {self.current_camera_index}")
            return
            
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.process_frame)
        
        # Update camera selection combo box
        if hasattr(self, 'camera_combo'):
            self.camera_combo.clear()
            for i, cam_idx in enumerate(self.available_cameras):
                self.camera_combo.addItem(f"Camera {cam_idx} {'(Front)' if i == 0 else '(Back)'}", cam_idx)
            self.camera_combo.setCurrentIndex(self.available_cameras.index(self.current_camera_index))
    
    def switch_camera(self, index):
        if not hasattr(self, 'available_cameras') or index < 0:
            return
            
        camera_idx = self.camera_combo.itemData(index)
        if camera_idx == self.current_camera_index:
            return
            
        # Stop current processing
        self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            
        # Switch to new camera
        self.current_camera_index = camera_idx
        self.setup_webcam()
        self.start_processing()
        self.statusBar().showMessage(f"Switched to camera {camera_idx}")
        
    def start_processing(self):
        self.timer.start(30)
        
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Hand detection ROI
        roi_size = 300
        roi_x = w//2 - roi_size//2
        roi_y = h//2 - roi_size//2
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x+roi_size, roi_y+roi_size), (0, 255, 0), 2)
        
        # Process hand region
        hand_roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        if hand_roi.size == 0:
            return
            
        # Preprocess for gesture recognition
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(hand_roi, [largest_contour], -1, (0, 0, 255), 2)
            
            # Resize and prepare for network input
            resized = cv2.resize(threshold, (16, 16), interpolation=cv2.INTER_AREA)
            network_input = resized.flatten() / 255.0
            
            # Update network state
            for i, val in enumerate(network_input):
                self.network.state[self.input_neurons[i]] = val * 100
                
            # Forward pass
            self.backprop_learner.forward_pass(network_input)
            
            # Update UI
            max_activation = 0
            recognized_gesture = ""
            for gesture, neuron in zip(self.gestures, self.output_neurons):
                activation = self.network.state[neuron]
                self.activation_bars[gesture].setValue(int(activation))
                if activation > max_activation:
                    max_activation = activation
                    recognized_gesture = gesture.capitalize()
                    
            if max_activation > 50:
                self.gesture_label.setText(recognized_gesture)
        
        # Display images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))
        
        # Display ROI
        roi_display = cv2.cvtColor(threshold, cv2.COLOR_GRAY2RGB)
        roi_display = cv2.resize(roi_display, (128, 128), interpolation=cv2.INTER_NEAREST)
        h, w, ch = roi_display.shape
        bytes_per_line = ch * w
        qt_roi = QtGui.QImage(roi_display.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.roi_label.setPixmap(QtGui.QPixmap.fromImage(qt_roi))
        
    def capture_gesture_sample(self):
        # Get current ROI image
        ret, frame = self.cap.read()
        if not ret:
            return
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        roi_size = 300
        roi_x = w//2 - roi_size//2
        roi_y = h//2 - roi_size//2
        hand_roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        
        if hand_roi.size == 0:
            return
            
        # Preprocess
        gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        resized = cv2.resize(threshold, (16, 16), interpolation=cv2.INTER_AREA)
        network_input = resized.flatten() / 255.0
        
        # Create target output
        target_gesture = self.gesture_combo.currentText().lower()
        target_output = [1.0 if g == target_gesture else 0.0 for g in self.gestures]
        
        # Add to training data
        self.training_data.append((network_input, target_output))
        self.sample_count_label.setText(f"Samples: {len(self.training_data)}")
        self.statusBar().showMessage(f"Captured sample for {target_gesture.capitalize()}")
        
    def train_network(self):
        if not self.training_data:
            QtWidgets.QMessageBox.warning(self, "Error", "No training samples collected")
            return
            
        # Create progress dialog
        progress = QtWidgets.QProgressDialog("Training Network...", "Cancel", 0, 1000, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.show()
        
        # Training callback
        def progress_callback(epoch, error):
            progress.setValue(epoch)
            QtWidgets.QApplication.processEvents()
            return not progress.wasCanceled()
        
        # Train network
        try:
            self.backprop_learner.train(
                self.training_data,
                epochs=1000,
                target_error_threshold=0.01,
                progress_callback=progress_callback
            )
            self.statusBar().showMessage("Training completed successfully")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Training Error", str(e))
        finally:
            progress.close()
            
    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        event.accept()

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = GestureRecognitionApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()