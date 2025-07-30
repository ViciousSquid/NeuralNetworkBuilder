import sys
import math
import random
import tkinter as tk

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Boid Class
class Boid:
    def __init__(self, x, y):
        self.position = [x, y]
        self.velocity = [random.uniform(-1, 1), random.uniform(-1, 1)]

        # Normalize velocity
        mag = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if mag > 0:
            self.velocity[0] /= mag
            self.velocity[1] /= mag

        self.max_speed = 2.5
        self.max_force = 0.05

    def update(self, boids):
        perception_radius = 50

        avg_pos = [0, 0]
        avg_vel = [0, 0]
        separation_force = [0, 0]
        total_in_perception = 0

        for other in boids:
            if other is self:
                continue
            distance = math.hypot(self.position[0] - other.position[0], self.position[1] - other.position[1])
            if distance < perception_radius:
                avg_pos[0] += other.position[0]
                avg_pos[1] += other.position[1]
                avg_vel[0] += other.velocity[0]
                avg_vel[1] += other.velocity[1]

                # Separation force calculation
                diff = [self.position[0] - other.position[0], self.position[1] - other.position[1]]
                if distance > 0:
                    diff[0] /= distance * distance  # Weight by inverse square of distance
                    diff[1] /= distance * distance
                separation_force[0] += diff[0]
                separation_force[1] += diff[1]

                total_in_perception += 1

        if total_in_perception > 0:
            # Cohesion vector
            avg_pos[0] /= total_in_perception
            avg_pos[1] /= total_in_perception
            cohesion_steer = [avg_pos[0] - self.position[0], avg_pos[1] - self.position[1]]

            # Alignment vector
            avg_vel[0] /= total_in_perception
            avg_vel[1] /= total_in_perception

            # Separation vector
            separation_force[0] /= total_in_perception
            separation_force[1] /= total_in_perception

            # Apply weights to forces
            cohesion_steer[0] *= 0.6
            cohesion_steer[1] *= 0.6
            avg_vel[0] *= 0.5
            avg_vel[1] *= 0.5
            separation_force[0] *= 1.2
            separation_force[1] *= 1.2

            # Combine forces
            accel = [cohesion_steer[0] + avg_vel[0] + separation_force[0], cohesion_steer[1] + avg_vel[1] + separation_force[1]]

            # Limit the force
            mag = math.sqrt(accel[0]**2 + accel[1]**2)
            if mag > self.max_force:
                accel[0] = (accel[0] / mag) * self.max_force
                accel[1] = (accel[1] / mag) * self.max_force

            self.velocity[0] += accel[0]
            self.velocity[1] += accel[1]

            # Limit the speed
            mag = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
            if mag > self.max_speed:
                self.velocity[0] = (self.velocity[0] / mag) * self.max_speed
                self.velocity[1] = (self.velocity[1] / mag) * self.max_speed

        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]
        self.borders()

    def borders(self):
        # Wrap the boid around the screen edges
        if self.position[0] < 0:
            self.position[0] = WIDTH
        if self.position[1] < 0:
            self.position[1] = HEIGHT
        if self.position[0] > WIDTH:
            self.position[0] = 0
        if self.position[1] > HEIGHT:
            self.position[1] = 0

# Main Application Window
class FlockingSimulation:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="black")
        self.canvas.pack()

        self.boids = [Boid(random.uniform(0, WIDTH), random.uniform(0, HEIGHT)) for _ in range(100)]

        self.update_simulation()

    def update_simulation(self):
        self.canvas.delete("all")
        for boid in self.boids:
            boid.update(self.boids)
            self.draw_boid(boid)
        self.root.after(20, self.update_simulation)  # Schedule the next update

    def draw_boid(self, boid):
        angle = math.degrees(math.atan2(boid.velocity[1], boid.velocity[0]))
        points = [
            boid.position[0] + 10 * math.cos(math.radians(angle)),
            boid.position[1] + 10 * math.sin(math.radians(angle)),
            boid.position[0] + 5 * math.cos(math.radians(angle + 150)),
            boid.position[1] + 5 * math.sin(math.radians(angle + 150)),
            boid.position[0] + 5 * math.cos(math.radians(angle - 150)),
            boid.position[1] + 5 * math.sin(math.radians(angle - 150))
        ]
        self.canvas.create_polygon(points, fill="cyan")

# Main function
def main():
    root = tk.Tk()
    root.title("Flocking / Swarm AI Simulation")
    app = FlockingSimulation(root)
    root.mainloop()

if __name__ == "__main__":
    main()
