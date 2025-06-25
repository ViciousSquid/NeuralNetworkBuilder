import sys
import os
import random
from PyQt5 import QtWidgets, QtCore, QtGui

# --- Add Project Root to sys.path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from NeuralNetwork.core import Network
from NeuralNetwork.learning import BackpropNetwork # We need the backprop learner

# --- Game Object Classes ---
class Paddle(QtWidgets.QGraphicsRectItem):
    def __init__(self, x, y):
        super().__init__(0, 0, 10, 80) # width, height
        self.setPos(x, y)
        self.setBrush(QtGui.QColor("white"))
        self.speed = 10

class Ball(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, x, y):
        super().__init__(0, 0, 15, 15)
        self.setPos(x, y)
        self.setBrush(QtGui.QColor("white"))
        self.dx = random.choice([-5, 5])
        self.dy = random.uniform(-4, 4)

# --- Main Game Window ---
class PongGame(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pong AI")
        self.setGeometry(200, 200, 800, 600)

        self.view = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QtGui.QColor("black"))
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setCentralWidget(self.view)
        
        self.game_width = 800
        self.game_height = 600
        self.scene.setSceneRect(0, 0, self.game_width, self.game_height)
        
        # Game objects
        self.player = Paddle(20, self.game_height / 2 - 40)
        self.ai_paddle = Paddle(self.game_width - 30, self.game_height / 2 - 40)
        self.ball = Ball(self.game_width / 2 - 7.5, self.game_height / 2 - 7.5)
        self.scene.addItem(self.player)
        self.scene.addItem(self.ai_paddle)
        self.scene.addItem(self.ball)

        # AI Setup
        self.ai_network = self.create_ai_brain()
        self.backprop_learner = BackpropNetwork(self.ai_network, learning_rate=0.2, momentum_factor=0.1)
        self.backprop_learner.set_layers([['ball_y', 'ball_dy', 'paddle_y'], ['h1', 'h2', 'h3', 'h4'], ['output']])
        self.train_ai()

        # Game Loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(16) # ~60 FPS
        
    def create_ai_brain(self):
        """ Create the neural network for the AI paddle. """
        net = Network()
        inputs = ['ball_y', 'ball_dy', 'paddle_y']
        hidden = ['h1', 'h2', 'h3', 'h4']
        outputs = ['output'] # Single output: move direction (-1 to 1)

        for name in inputs: net.add_neuron(name, n_type='input')
        for name in hidden: net.add_neuron(name, n_type='hidden')
        for name in outputs: net.add_neuron(name, n_type='output')
        
        # Connect layers fully
        for i_name in inputs:
            for h_name in hidden:
                net.connect(i_name, h_name, random.uniform(-0.5, 0.5))
        for h_name in hidden:
            for o_name in outputs:
                net.connect(h_name, o_name, random.uniform(-0.5, 0.5))
        return net
        
    def train_ai(self):
        """ Generate training data from a 'perfect' algorithm and train the network. """
        print("Training AI...")
        training_data = []
        # Generate 2000 examples
        for _ in range(2000):
            # Simulate random game states
            ball_y = random.uniform(0, self.game_height)
            ball_dy = random.uniform(-5, 5)
            paddle_y = random.uniform(0, self.game_height - 80)
            
            # 'Perfect' algorithm: if ball is above paddle, move up. If below, move down.
            perfect_move = 0
            if ball_y < paddle_y + 40: # Center of paddle
                perfect_move = 0.0 # Move Up
            else:
                perfect_move = 1.0 # Move Down

            inputs = [ball_y / self.game_height, ball_dy / 5, paddle_y / self.game_height]
            outputs = [perfect_move]
            training_data.append((inputs, outputs))
            
        # Train the network
        self.backprop_learner.train(training_data, epochs=100, target_error_threshold=0.01)
        print("AI Training complete.")
        
    def game_loop(self):
        # --- Ball Movement & Collision ---
        self.ball.setPos(self.ball.x() + self.ball.dx, self.ball.y() + self.ball.dy)

        # Wall collision (top/bottom)
        if self.ball.y() <= 0 or self.ball.y() >= self.game_height - 15:
            self.ball.dy *= -1
            
        # Paddle collision
        if self.ball.collidesWithItem(self.player) or self.ball.collidesWithItem(self.ai_paddle):
            self.ball.dx *= -1.1 # Increase speed
            self.ball.dy += random.uniform(-0.5, 0.5) # Add slight angle change

        # Score
        if self.ball.x() < 0 or self.ball.x() > self.game_width:
            self.ball.setPos(self.game_width / 2 - 7.5, self.game_height / 2 - 7.5)
            self.ball.dx = random.choice([-5, 5])
            self.ball.dy = random.uniform(-4, 4)

        # --- AI Movement ---
        self.update_ai_paddle()

    def update_ai_paddle(self):
        # Provide normalized inputs to the network
        inputs = [
            self.ball.y() / self.game_height,
            self.ball.dy / 5.0, # Normalize by a reasonable max speed
            self.ai_paddle.y() / self.game_height
        ]
        
        # Get the network's decision
        output = self.backprop_learner.forward_pass(inputs)[0] # Get the single output value
        
        # Interpret the output
        if output < 0.45: # Move up
             self.ai_paddle.setY(max(0, self.ai_paddle.y() - self.ai_paddle.speed))
        elif output > 0.55: # Move down
             self.ai_paddle.setY(min(self.game_height - 80, self.ai_paddle.y() + self.ai_paddle.speed))
        # else: stay still (if output is around 0.5)

    def mouseMoveEvent(self, event):
        # Player control
        self.player.setY(event.y() - 40)
        # Keep paddle on screen
        if self.player.y() < 0: self.player.setY(0)
        if self.player.y() > self.game_height - 80: self.player.setY(self.game_height - 80)
        
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)


def main():
    app = QtWidgets.QApplication(sys.argv)
    game = PongGame()
    game.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()