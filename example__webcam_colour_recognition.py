import sys
import os
import cv2 
import numpy as np 
import time
import random

from PyQt5 import QtWidgets, QtCore, QtGui

# Corrected path manipulation
grandparent_dir = os.path.dirname(os.path.abspath(__file__))
if grandparent_dir not in sys.path:
    sys.path.insert(0, grandparent_dir)

from NeuralNetwork.core import Network, Config
from NeuralNetwork.visualization import NetworkVisualization
from NeuralNetwork.learning import BackpropNetwork

class WebcamColorApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.network = None 
        self.backprop_learner = None
        self.training_data_normalized = [] 
        self.setup_network() 
        self.setup_ui()      
        self.setup_webcam()
        self.start_processing()
        self.current_rgb_center = (0,0,0)

    def setup_ui(self):
        self.setWindowTitle("Neural Network Colour Recognition (Backprop)")
        self.setGeometry(100, 100, 1100, 700)
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QtWidgets.QHBoxLayout(self.central_widget)

        # Create a splitter for resizable panes
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        # Webcam Label (Left Pane)
        self.webcam_label = QtWidgets.QLabel("Initializing Webcam...")
        self.webcam_label.setMinimumSize(640, 480); self.webcam_label.setAlignment(QtCore.Qt.AlignCenter)
        self.webcam_label.setStyleSheet("background-color: #333; color: white; font-size: 16px;")
        
        # Right Panel (Controls and Visualization)
        right_panel_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel_widget)
        self.vis = NetworkVisualization(self.network, self)
        self.vis.setMinimumSize(400, 300)
        right_layout.addWidget(self.vis)
        
        controls_info_group = QtWidgets.QGroupBox("Interaction & Training")
        controls_info_layout = QtWidgets.QVBoxLayout(controls_info_group)
        self.rgb_display_label = QtWidgets.QLabel("Center RGB: (0, 0, 0)")
        self.rgb_display_label.setStyleSheet("font-size: 14px; padding: 5px; background-color: #f0f0f0; border-radius: 3px;")
        controls_info_layout.addWidget(self.rgb_display_label)
        self.color_preview = QtWidgets.QLabel()
        self.color_preview.setMinimumSize(50,20); self.color_preview.setAutoFillBackground(True)
        controls_info_layout.addWidget(QtWidgets.QLabel("Sampled Colour Preview:")); controls_info_layout.addWidget(self.color_preview)
        
        sampling_group = QtWidgets.QGroupBox("Colour Sampling for Training")
        sampling_layout = QtWidgets.QFormLayout(sampling_group)
        self.color_target_combo = QtWidgets.QComboBox()
        self.color_target_combo.addItems([name.replace("_out","").capitalize() for name in self.output_neuron_names_for_sampling])
        self.color_target_combo.setToolTip("Select the colour category for the current sample.")
        sampling_layout.addRow("Assign Sample to:", self.color_target_combo)
        self.sample_color_btn = QtWidgets.QPushButton("Take Color Sample")
        self.sample_color_btn.clicked.connect(self.take_color_sample_for_training)
        self.sample_color_btn.setToolTip("Capture the current center colour and assign it to the selected category for training.")
        sampling_layout.addRow(self.sample_color_btn)
        self.training_samples_count_label = QtWidgets.QLabel(f"Samples: {len(self.training_data_normalized)}")
        sampling_layout.addRow(self.training_samples_count_label)
        controls_info_layout.addWidget(sampling_group)
        
        self.train_network_btn = QtWidgets.QPushButton("Train Network on Samples")
        self.train_network_btn.clicked.connect(self.train_network_on_samples)
        self.train_network_btn.setToolTip("Train the network using all collected colour samples.")
        controls_info_layout.addWidget(self.train_network_btn)
        
        activation_group = QtWidgets.QGroupBox("Output Neuron Activations (0-1)")
        activation_form_layout = QtWidgets.QFormLayout(activation_group)
        self.activation_labels = {}
        for name in self.output_neuron_names_for_sampling:
            label = QtWidgets.QLabel("0.000"); label.setStyleSheet("font-weight: bold;")
            self.activation_labels[name] = label
            activation_form_layout.addRow(f"{name.replace('_out','').capitalize()}:", label)
        controls_info_layout.addWidget(activation_group)
        
        self.dominant_color_label = QtWidgets.QLabel("Recognized: None")
        self.dominant_color_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px; background-color: #e0e0e0; border-radius: 3px;")
        self.dominant_color_label.setAlignment(QtCore.Qt.AlignCenter)
        controls_info_layout.addWidget(self.dominant_color_label)
        
        self.reset_network_btn = QtWidgets.QPushButton("Reset Network Weights & Samples")
        self.reset_network_btn.clicked.connect(self.reset_network_action)
        controls_info_layout.addWidget(self.reset_network_btn)
        
        right_layout.addWidget(controls_info_group)
        
        # Add panes to the splitter
        self.splitter.addWidget(self.webcam_label)
        self.splitter.addWidget(right_panel_widget)
        
        # Set initial sizes for the panes
        self.splitter.setSizes([450, 650])
        
        # Add splitter to the main layout
        main_layout.addWidget(self.splitter)
        
        self.statusBar().showMessage("Ready. Webcam starting...")
    
    def setup_network(self):
        self.network = Network()
        self.network.config.hebbian['learning_interval'] = float('inf')
        self.network.set_neurogenesis_enabled(False)
        input_names = ["cam_R", "cam_G", "cam_B"]
        hidden_names = [f"hidden_{i+1}" for i in range(5)]
        self.output_neuron_names_for_sampling = ["red_out", "green_out", "blue_out", "yellow_out", "other_out"]
        self.network.add_neuron(input_names[0],value=0,position=(100,100),n_type="input",attributes={'color':(255,100,100)})
        self.network.add_neuron(input_names[1],value=0,position=(100,200),n_type="input",attributes={'color':(100,255,100)})
        self.network.add_neuron(input_names[2],value=0,position=(100,300),n_type="input",attributes={'color':(100,100,255)})
        for i,name in enumerate(hidden_names): self.network.add_neuron(name,value=50,position=(280,70+i*70),n_type="hidden")
        output_colors_map={"red_out":(220,50,50),"green_out":(50,200,50),"blue_out":(50,50,220),"yellow_out":(220,220,50),"other_out":(128,128,128)}
        for i,name in enumerate(self.output_neuron_names_for_sampling): self.network.add_neuron(name,value=50,position=(450,70+i*70),n_type="output",attributes={'color':output_colors_map[name]})
        for in_n in input_names:
            for hid_n in hidden_names: self.network.connect(in_n,hid_n,random.uniform(-0.3,0.3))
        for hid_n in hidden_names:
            for out_n in self.output_neuron_names_for_sampling: self.network.connect(hid_n,out_n,random.uniform(-0.3,0.3))
        self.backprop_learner = BackpropNetwork(self.network,learning_rate=0.15,momentum_factor=0.2)
        self.backprop_learner.set_layers([input_names,hidden_names,self.output_neuron_names_for_sampling])
        print("Webcam network initialized with BackpropLearner.")

    def randomize_network_weights(self):
        if self.network:
            for conn_key in self.network.connections: self.network.connections[conn_key].set_weight(random.uniform(-0.3,0.3))
            self.vis.update(); self.statusBar().showMessage("Network weights randomized.")
            if self.backprop_learner: self.backprop_learner.previous_weight_updates={}

    def reset_network_action(self):
        if QtWidgets.QMessageBox.question(self,"Reset","Reset weights & clear samples?",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No,QtWidgets.QMessageBox.No)==QtWidgets.QMessageBox.Yes:
            self.randomize_network_weights()
            self.training_data_normalized=[]; self.training_samples_count_label.setText(f"Samples: {len(self.training_data_normalized)}")
            self.statusBar().showMessage("Network weights randomized & samples cleared.")

    def setup_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened(): self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened(): QtWidgets.QMessageBox.critical(self,"Webcam Error","Could not open webcam."); self.webcam_label.setText("Webcam Error"); return
        self.timer = QtCore.QTimer(); self.timer.timeout.connect(self.process_webcam_frame)
        self.statusBar().showMessage("Webcam initialized.")
    
    def start_processing(self):
        if hasattr(self,'cap') and self.cap and self.cap.isOpened(): self.timer.start(66)
        else: self.statusBar().showMessage("Webcam not available.")
    
    def process_webcam_frame(self):
        if not hasattr(self,'cap')or not self.cap or not self.cap.isOpened(): return
        ret,frame=self.cap.read()
        if not ret: self.statusBar().showMessage("Error reading frame.",2000); return
        frame=cv2.flip(frame,1)
        h,w,_=frame.shape;cx,cy=w//2,h//2;ssize=30
        x1,y1,x2,y2=cx-ssize//2,cy-ssize//2,cx+ssize//2,cy+ssize//2
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        region=frame[y1:y2,x1:x2]
        if region.size==0: return
        avg_bgr=region.mean(axis=(0,1)); self.current_rgb_center=(int(avg_bgr[2]),int(avg_bgr[1]),int(avg_bgr[0]))
        r,g,b=self.current_rgb_center
        self.rgb_display_label.setText(f"Center RGB: ({r},{g},{b})")
        pal=self.color_preview.palette();pal.setColor(QtGui.QPalette.Window,QtGui.QColor(r,g,b));self.color_preview.setPalette(pal)
        norm_inputs=[r/255.0,g/255.0,b/255.0]
        output_activations_normalized=self.backprop_learner.forward_pass(norm_inputs)
        max_act=-1;dominant_color_name="Uncertain"
        for i,out_name in enumerate(self.output_neuron_names_for_sampling):
            act_norm=output_activations_normalized[i]
            self.activation_labels[out_name].setText(f"{act_norm:.3f}")
            if act_norm>max_act:max_act=act_norm;dominant_color_name=out_name.replace("_out","").capitalize()
        if max_act>0.6:
            self.dominant_color_label.setText(f"Recognized: {dominant_color_name}")
            dom_col_tuple=self.network.neurons[dominant_color_name.lower()+"_out"].attributes.get('color',(200,200,200))
            self.dominant_color_label.setStyleSheet(f"font-size:16px;font-weight:bold;padding:5px;border-radius:3px;background-color:rgb({dom_col_tuple[0]},{dom_col_tuple[1]},{dom_col_tuple[2]});color:{'black'if sum(dom_col_tuple)>384 else'white'};")
        else:
            self.dominant_color_label.setText("Recognized: Uncertain")
            self.dominant_color_label.setStyleSheet("font-size:16px;font-weight:bold;padding:5px;background-color:#e0e0e0;border-radius:3px;")
        self.vis.update()
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        h,w,ch=rgb_frame.shape;bytesPerLine=ch*w
        qtImg=QtGui.QImage(rgb_frame.data,w,h,bytesPerLine,QtGui.QImage.Format_RGB888)
        pixmap=QtGui.QPixmap.fromImage(qtImg)
        self.webcam_label.setPixmap(pixmap.scaled(self.webcam_label.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))

    def take_color_sample_for_training(self):
        target_color_label=self.color_target_combo.currentText().lower()
        target_color_name=f"{target_color_label}_out" # Construct full neuron name
        if target_color_name not in self.output_neuron_names_for_sampling:
            QtWidgets.QMessageBox.warning(self,"Sampling Error","Invalid target color.");return
        r,g,b=self.current_rgb_center
        norm_rgb_in=[r/255.0,g/255.0,b/255.0]
        target_out_norm=[0.0]*len(self.output_neuron_names_for_sampling)
        try:
            target_idx=self.output_neuron_names_for_sampling.index(target_color_name)
            target_out_norm[target_idx]=1.0
        except ValueError:
            QtWidgets.QMessageBox.warning(self,"Sampling Error",f"Target neuron {target_color_name} not found.");return
        self.training_data_normalized.append((norm_rgb_in,target_out_norm))
        self.training_samples_count_label.setText(f"Samples: {len(self.training_data_normalized)}")
        self.statusBar().showMessage(f"Sampled RGB ({r},{g},{b}) as '{target_color_label.capitalize()}'. Total: {len(self.training_data_normalized)}",5000)

    def train_network_on_samples(self):
        if not self.training_data_normalized:QtWidgets.QMessageBox.information(self,"Training","No samples.");return
        epochs=200;target_error=0.01
        self.setEnabled(False)
        progress=QtWidgets.QProgressDialog("Training network...","Cancel",0,epochs,self)
        progress.setWindowModality(QtCore.Qt.WindowModal);progress.show()
        epoch_errors_hist=[]
        def cb(epoch,avg_error):
            progress.setValue(epoch+1);progress.setLabelText(f"Epoch {epoch+1}/{epochs},AvgErr:{avg_error:.5f}")
            QtWidgets.QApplication.processEvents();return not progress.wasCanceled()
        try:
            epoch_errors_hist=self.backprop_learner.train(self.training_data_normalized,epochs=epochs,target_error_threshold=target_error,progress_callback=cb)
            final_err=epoch_errors_hist[-1] if epoch_errors_hist else float('inf')
            if progress.wasCanceled():self.statusBar().showMessage("Training cancelled.")
            else:progress.setValue(epochs);self.statusBar().showMessage(f"Training complete. Final avg err:{final_err:.5f}",10000)
        except Exception as e:QtWidgets.QMessageBox.critical(self,"Training Error",f"Error: {e}");self.statusBar().showMessage(f"Training failed:{e}")
        finally:self.setEnabled(True);progress.close();self.vis.update()

    def closeEvent(self,event):
        self.timer.stop()
        if hasattr(self,'cap')and self.cap and self.cap.isOpened():self.cap.release()
        cv2.destroyAllWindows();event.accept()

def main():
    app=QtWidgets.QApplication(sys.argv)
    if "Fusion" in QtWidgets.QStyleFactory.keys():app.setStyle(QtWidgets.QStyleFactory.create("Fusion"))
    window=WebcamColorApp();window.show();sys.exit(app.exec_())

if __name__=="__main__":
    main()