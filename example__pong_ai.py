import sys
import os
import random
import tkinter as tk

# --- Add Project Root to sys.path ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from NeuralNetwork.core import Network
from NeuralNetwork.learning import BackpropNetwork # We need the backprop learner

# --- Main Game Window ---
class PongGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Pong AI (Tkinter)")

        # Game constants
        self.game_width = 800
        self.game_height = 600
        self.paddle_width = 10
        self.paddle_height = 80
        self.ball_size = 15
        self.paddle_speed = 6
        
        self.canvas = tk.Canvas(root, width=self.game_width, height=self.game_height, bg="black")
        self.canvas.pack()

        # Game objects
        self.player = self.create_paddle(20, self.game_height / 2 - self.paddle_height / 2)
        self.ai_paddle = self.create_paddle(self.game_width - 30, self.game_height / 2 - self.paddle_height / 2)
        self.ball = self.create_ball(self.game_width / 2 - self.ball_size / 2, self.game_height / 2 - self.ball_size / 2)
        
        # Ball velocity
        self.ball_dx = random.choice([-5, 5])
        self.ball_dy = random.uniform(-4, 4)
        
        # Player movement state
        self.player_dy = 0
        
        # Keyboard bindings
        self.root.bind("<KeyPress-Up>", self.move_up)
        self.root.bind("<KeyPress-Down>", self.move_down)
        self.root.bind("<KeyRelease-Up>", self.stop_move)
        self.root.bind("<KeyRelease-Down>", self.stop_move)

        # AI Setup
        self.ai_network = self.create_ai_brain()
        self.backprop_learner = BackpropNetwork(self.ai_network, learning_rate=0.2, momentum_factor=0.1)
        self.backprop_learner.set_layers([['ball_y', 'ball_dy', 'paddle_y'], ['h1', 'h2', 'h3', 'h4'], ['output']])
        self.train_ai()

        # Game Loop
        self.game_loop()

    def create_paddle(self, x, y):
        return self.canvas.create_rectangle(x, y, x + self.paddle_width, y + self.paddle_height, fill="white", outline="white")

    def create_ball(self, x, y):
        return self.canvas.create_oval(x, y, x + self.ball_size, y + self.ball_size, fill="white", outline="white")

    # --- Player Controls ---
    def move_up(self, event):
        self.player_dy = -self.paddle_speed

    def move_down(self, event):
        self.player_dy = self.paddle_speed

    def stop_move(self, event):
        self.player_dy = 0

    # --- AI Methods ---
    def create_ai_brain(self):
        """ Create the neural network for the AI paddle. """
        net = Network()
        inputs = ['ball_y', 'ball_dy', 'paddle_y']
        hidden = ['h1', 'h2', 'h3', 'h4']
        outputs = ['output'] # Single output: move direction (-1 to 1)

        for name in inputs: net.add_neuron(name, 0.0, (0, 0), n_type='input')
        for name in hidden: net.add_neuron(name, 0.0, (0, 0), n_type='hidden')
        for name in outputs: net.add_neuron(name, 0.0, (0, 0), n_type='output')
        
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
        print("* Generating 2000 example games of pong...")
        print("Training AI on the data..  [PLEASE WAIT]")
        training_data = []
        # Generate 2000 examples
        for _ in range(2000):
            # Simulate random game states
            ball_y = random.uniform(0, self.game_height)
            ball_dy = random.uniform(-5, 5)
            paddle_y = random.uniform(0, self.game_height - self.paddle_height)
            
            # 'Perfect' algorithm: if ball is above paddle, move up. If below, move down.
            perfect_move = 0.0 # Move Up
            if ball_y > paddle_y + self.paddle_height / 2:
                perfect_move = 1.0 # Move Down

            inputs = [ball_y / self.game_height, ball_dy / 5, paddle_y / self.game_height]
            outputs = [perfect_move]
            training_data.append((inputs, outputs))
            
        # Train the network
        self.backprop_learner.train(training_data, epochs=100, target_error_threshold=0.01)
        print("AI Training complete.")
        print("Use cursor UP and DN to play")
    
    def update_ai_paddle(self):
        """ Use the neural network to decide the AI paddle's movement. """
        ball_pos = self.canvas.coords(self.ball)
        ai_pos = self.canvas.coords(self.ai_paddle)
        
        ball_y = ball_pos[1]
        ai_y = ai_pos[1]

        # Provide normalized inputs to the network
        inputs = [
            ball_y / self.game_height,
            self.ball_dy / 5.0, # Normalize by a reasonable max speed
            ai_y / self.game_height
        ]
        
        # Get the network's decision
        output = self.backprop_learner.forward_pass(inputs)[0]
        
        # Interpret the output to move the paddle
        ai_move = 0
        if output < 0.45: # Move up
            ai_move = -self.paddle_speed
        elif output > 0.55: # Move down
            ai_move = self.paddle_speed

        # Apply movement and check boundaries
        self.canvas.move(self.ai_paddle, 0, ai_move)
        pos = self.canvas.coords(self.ai_paddle)
        if pos[1] < 0:
            self.canvas.move(self.ai_paddle, 0, -pos[1])
        elif pos[3] > self.game_height:
            self.canvas.move(self.ai_paddle, 0, self.game_height - pos[3])

    # --- Main Game Loop ---
    def game_loop(self):
        # Move player paddle
        self.canvas.move(self.player, 0, self.player_dy)
        player_pos = self.canvas.coords(self.player)
        if player_pos[1] < 0:
            self.canvas.move(self.player, 0, -player_pos[1])
        elif player_pos[3] > self.game_height:
            self.canvas.move(self.player, 0, self.game_height - player_pos[3])
            
        # Move ball
        self.canvas.move(self.ball, self.ball_dx, self.ball_dy)
        ball_pos = self.canvas.coords(self.ball)
        
        # Wall collision (top/bottom)
        if ball_pos[1] <= 0 or ball_pos[3] >= self.game_height:
            self.ball_dy *= -1
            
        # Paddle collision
        player_coords = self.canvas.coords(self.player)
        ai_coords = self.canvas.coords(self.ai_paddle)
        # Check for overlap with either paddle
        if (self.ball_dx < 0 and ball_pos[0] < player_coords[2] and ball_pos[2] > player_coords[0] and ball_pos[1] < player_coords[3] and ball_pos[3] > player_coords[1]) or \
           (self.ball_dx > 0 and ball_pos[0] < ai_coords[2] and ball_pos[2] > ai_coords[0] and ball_pos[1] < ai_coords[3] and ball_pos[3] > ai_coords[1]):
            self.ball_dx *= -1.1 # Increase speed
            self.ball_dy += random.uniform(-0.5, 0.5) # Add slight angle change

        # Score and reset
        if ball_pos[0] < 0 or ball_pos[2] > self.game_width:
            self.canvas.coords(self.ball, self.game_width/2 - self.ball_size/2, self.game_height/2 - self.ball_size/2, 
                                         self.game_width/2 + self.ball_size/2, self.game_height/2 + self.ball_size/2)
            self.ball_dx = random.choice([-5, 5])
            self.ball_dy = random.uniform(-4, 4)

        # Update AI
        self.update_ai_paddle()
        
        # Repeat the loop
        self.root.after(16, self.game_loop) # ~60 FPS

def main():
    root = tk.Tk()
    game = PongGame(root)
    root.mainloop()

if __name__ == "__main__":
    main()