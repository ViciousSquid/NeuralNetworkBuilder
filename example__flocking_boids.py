import sys
import os
import math
import random
from PyQt5 import QtWidgets, QtCore, QtGui

# --- Add Project Root to sys.path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from NeuralNetwork.core import Network, Config

# --- Boid Class ---
class Boid:
    """ A single agent in the flock. """
    def __init__(self, x, y, width, height):
        self.position = QtCore.QPointF(x, y)
        self.velocity = QtCore.QPointF(random.uniform(-1, 1), random.uniform(-1, 1))
        # Normalize the velocity
        mag = math.sqrt(self.velocity.x()**2 + self.velocity.y()**2)
        if mag > 0:
            self.velocity = QtCore.QPointF(self.velocity.x()/mag, self.velocity.y()/mag)
        self.width = width  # Simulation space width
        self.height = height # Simulation space height
        self.max_speed = 2.5
        self.max_force = 0.05
        
        # Each boid has its own simple neural network
        self.network = self.create_brain()

    def create_brain(self):
        """ Creates a simple neural network for the boid. """
        net = Network()
        # Inputs: avg_flock_x, avg_flock_y, avg_heading_x, avg_heading_y, avg_separation_x, avg_separation_y
        inputs = ['avg_flock_x', 'avg_flock_y', 'avg_heading_x', 'avg_heading_y', 'sep_x', 'sep_y']
        # Outputs: acceleration_x, acceleration_y
        outputs = ['accel_x', 'accel_y']
        
        # Default position for neurons (not used visually in this simulation)
        default_pos = (0, 0)
        
        for name in inputs:
            net.add_neuron(name, 50.0, position=default_pos, n_type='input')
        for name in outputs:
            net.add_neuron(name, 50.0, position=default_pos, n_type='output')
            
        # A simple brain: directly connect inputs to outputs
        for i_name in inputs:
            for o_name in outputs:
                # Weights are tuned to produce classic flocking behavior
                weight = 0
                if (i_name.startswith('avg_flock') and o_name.endswith(i_name[-1])): # Cohesion
                    weight = 0.6
                elif (i_name.startswith('avg_heading') and o_name.endswith(i_name[-1])): # Alignment
                    weight = 0.5
                elif (i_name.startswith('sep') and o_name.endswith(i_name[-1])): # Separation
                    weight = 1.2 # Strongest influence
                net.connect(i_name, o_name, weight)
        return net

    def update(self, boids):
        """ Update the boid's state based on the flock. """
        perception_radius = 50
        
        # --- Calculate local flock dynamics ---
        avg_pos = QtCore.QPointF(0, 0)
        avg_vel = QtCore.QPointF(0, 0)
        separation_force = QtCore.QPointF(0, 0)
        total_in_perception = 0

        for other in boids:
            if other is self:
                continue
            distance = math.hypot(self.position.x() - other.position.x(), self.position.y() - other.position.y())
            if distance < perception_radius:
                avg_pos += other.position
                avg_vel += other.velocity
                
                # Separation force calculation
                diff = self.position - other.position
                if distance > 0:
                    diff /= (distance * distance) # Weight by inverse square of distance
                separation_force += diff
                
                total_in_perception += 1

        # --- Feed inputs into the neural network ---
        if total_in_perception > 0:
            # Cohesion vector
            avg_pos /= total_in_perception
            cohesion_steer = (avg_pos - self.position)
            
            # Alignment vector
            avg_vel /= total_in_perception
            
            # Separation vector
            separation_force /= total_in_perception

            # Set network inputs (normalized)
            self.network.state['avg_flock_x'] = (cohesion_steer.x() / perception_radius) * 100
            self.network.state['avg_flock_y'] = (cohesion_steer.y() / perception_radius) * 100
            self.network.state['avg_heading_x'] = avg_vel.x() * 100
            self.network.state['avg_heading_y'] = avg_vel.y() * 100
            self.network.state['sep_x'] = separation_force.x() * 100
            self.network.state['sep_y'] = separation_force.y() * 100
        else:
            # No neighbors, no input
            for name in self.network.neurons: 
                self.network.state[name] = 0

        # Propagate through the network to get desired acceleration
        self.network.propagate_activation()
        
        # --- Apply the network's output ---
        accel = QtCore.QPointF(
            self.network.state.get('accel_x', 0) / 100.0,  # Scale back down
            self.network.state.get('accel_y', 0) / 100.0   # Scale back down
        )
        
        # Limit the force
        mag = math.sqrt(accel.x()**2 + accel.y()**2)
        if mag > self.max_force and mag > 0:
            accel = QtCore.QPointF(accel.x()/mag * self.max_force, 
                                  accel.y()/mag * self.max_force)

        self.velocity += accel
        
        # Limit the speed
        mag = math.sqrt(self.velocity.x()**2 + self.velocity.y()**2)
        if mag > self.max_speed and mag > 0:
            self.velocity = QtCore.QPointF(self.velocity.x()/mag * self.max_speed, 
                                         self.velocity.y()/mag * self.max_speed)
            
        self.position += self.velocity
        self.borders()

    def borders(self):
        """ Wrap the boid around the screen edges. """
        if self.position.x() < 0: self.position.setX(self.width)
        if self.position.y() < 0: self.position.setY(self.height)
        if self.position.x() > self.width: self.position.setX(0)
        if self.position.y() > self.height: self.position.setY(0)

# --- Main Application Window ---
class FlockingWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flocking / Swarm AI Simulation")
        self.setGeometry(100, 100, 1024, 768)

        self.view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.setCentralWidget(self.view)
        
        self.flock = []
        self.boid_items = []
        
        self.setup_flock(100)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.timer.start(20) # ~50 FPS

    def setup_flock(self, num_boids):
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        boid_color = QtGui.QColor("cyan")
        
        for _ in range(num_boids):
            boid = Boid(random.uniform(0, self.width()), 
                        random.uniform(0, self.height()), 
                        self.width(), 
                        self.height())
            self.flock.append(boid)
            
            # Create the visual representation (a triangle)
            polygon = QtGui.QPolygonF([
                QtCore.QPointF(-5, -5),
                QtCore.QPointF(10, 0),
                QtCore.QPointF(-5, 5)
            ])
            item = QtWidgets.QGraphicsPolygonItem(polygon)
            item.setBrush(boid_color)
            self.scene.addItem(item)
            self.boid_items.append(item)

    def update_simulation(self):
        for boid in self.flock:
            boid.update(self.flock)
            
        self.draw_flock()

    def draw_flock(self):
        for i, boid in enumerate(self.flock):
            item = self.boid_items[i]
            item.setPos(boid.position)
            # Rotate the triangle to match the boid's velocity direction
            angle = math.degrees(math.atan2(boid.velocity.y(), boid.velocity.x()))
            item.setRotation(angle)
            
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        for boid in self.flock:
            boid.width = self.width()
            boid.height = self.height()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = FlockingWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()