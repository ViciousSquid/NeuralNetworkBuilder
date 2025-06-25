# NeuralNetwork/visualization.py
import math
from PyQt5 import QtCore, QtGui, QtWidgets

class NetworkVisualization(QtWidgets.QWidget):
    # Signals for robust communication with the main GUI
    canvasClicked = QtCore.pyqtSignal(QtGui.QMouseEvent)
    canvasMoved = QtCore.pyqtSignal(QtGui.QMouseEvent)
    canvasReleased = QtCore.pyqtSignal(QtGui.QMouseEvent)

    # Original signals
    neuronClicked = QtCore.pyqtSignal(str)
    selectionChanged = QtCore.pyqtSignal(set)

    def __init__(self, network, parent=None):
        super().__init__(parent)
        self.network = network
        self.layers_data = {}
        self.pan_offset = QtCore.QPointF(0, 0)
        self.zoom_factor = 1.0
        self.setMouseTracking(True)
        self.dragging = False
        self.last_mouse_pos = QtCore.QPoint()
        self.show_weights = True
        self.show_links = True
        self.current_mouse_mode = "select"
        self.selected_neurons = set()
        self.highlighted_neuron = None
        self.highlight_timer = QtCore.QTimer()
        self.highlight_timer.setSingleShot(True)
        self.highlight_timer.timeout.connect(self._clear_highlight)
        
        # FIX: Added state for dragging neurons
        self.dragged_neuron = None
        self.drag_offset = QtCore.QPointF(0,0)


    def set_zoom(self, factor):
        self.zoom_factor = factor
        self.update()

    def set_layers_data(self, layers):
        self.layers_data = layers
        self._update_layer_rects()
        self.update()

    def _update_layer_rects(self):
        # A helper to pre-calculate layer boundaries for drawing
        pass

    def _widget_to_logical(self, widget_pos):
        return (widget_pos - self.pan_offset) / self.zoom_factor

    def get_neuron_at_pos(self, widget_pos):
        logical_pos = self._widget_to_logical(widget_pos)
        for name, neuron in self.network.neurons.items():
            neuron_pos = QtCore.QPointF(*neuron.get_position())
            # Increased radius for easier clicking
            if (logical_pos - neuron_pos).manhattanLength() < 25: 
                return name
        return None

    def get_layer_at_pos(self, widget_pos):
        # Placeholder for layer selection logic
        return None

    def highlight_new_neuron(self, neuron_name, duration_sec):
        self.highlighted_neuron = neuron_name
        self.highlight_timer.start(int(duration_sec * 1000))
        self.update()

    def _clear_highlight(self):
        self.highlighted_neuron = None
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtCore.Qt.white)
        
        painter.translate(self.pan_offset)
        painter.scale(self.zoom_factor, self.zoom_factor)

        # Draw Connections
        if self.show_links:
            for (source, target), conn in self.network.connections.items():
                if source in self.network.neurons and target in self.network.neurons:
                    p1 = QtCore.QPointF(*self.network.neurons[source].get_position())
                    p2 = QtCore.QPointF(*self.network.neurons[target].get_position())
                    
                    color = QtGui.QColor(0, 255, 0) if conn.get_weight() > 0 else QtGui.QColor(255, 0, 0)
                    pen = QtGui.QPen(color, max(1, abs(conn.get_weight()) * 4))
                    painter.setPen(pen)
                    painter.drawLine(p1, p2)
                    
                    if self.show_weights:
                        mid_point = (p1 + p2) / 2
                        painter.setPen(QtCore.Qt.black)
                        painter.drawText(mid_point, f"{conn.get_weight():.2f}")

        # Draw Neurons
        for name, neuron in self.network.neurons.items():
            pos = QtCore.QPointF(*neuron.get_position())
            radius = 25
            
            color_tuple = neuron.attributes.get('color', (180, 180, 180))
            brush_color = QtGui.QColor(*color_tuple)
            
            if name in self.selected_neurons:
                pen = QtGui.QPen(QtCore.Qt.blue, 3)
            elif name == self.highlighted_neuron:
                 pen = QtGui.QPen(QtCore.Qt.yellow, 4)
            else:
                pen = QtGui.QPen(QtCore.Qt.black, 1)

            painter.setPen(pen)
            painter.setBrush(brush_color)
            painter.drawEllipse(pos, radius, radius)
            
            painter.setPen(QtCore.Qt.black)
            painter.drawText(pos - QtCore.QPointF(radius, -radius), name)
            
            activation = self.network.state.get(name, 0)
            painter.drawText(pos + QtCore.QPointF(-10, 5), f"{activation:.1f}")

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.last_mouse_pos = event.pos()
        if event.button() == QtCore.Qt.RightButton:
            self.dragging = True
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.ClosedHandCursor)
        elif event.button() == QtCore.Qt.LeftButton:
            self.canvasClicked.emit(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self.dragging:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += delta
            self.last_mouse_pos = event.pos()
            self.update()
        else:
            self.canvasMoved.emit(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.RightButton and self.dragging:
            self.dragging = False
            QtWidgets.QApplication.restoreOverrideCursor()
        else:
            self.canvasReleased.emit(event)