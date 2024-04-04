# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define parameters
beta = 1  # Transmission rate # Reduced transmission rate to slow down the spread
delta = 80  # Elimination rate of zombies by humans # Increased elimination rate of zombies by humans
gamma = 150  # Recovery rate of zombies back to humans # Increased recovery rate of zombies back to humans

 

# Define the system of differential equations
def extended_zombie_apocalypse(S, Z):
    dSdt = -beta * S * Z  # Rate of change of Susceptible Humans
    dZdt = beta * S * Z - delta * Z - gamma * Z  # Rate of change of Zombies
    return dSdt, dZdt

# Create a grid of values for S and Z
S_values, Z_values = np.meshgrid(np.linspace(0, 500, 30), np.linspace(0, 500, 30))

# Compute the vector field
dS, dZ = extended_zombie_apocalypse(S_values, Z_values)

# Normalize vectors for better visualization
N = np.sqrt(dS**2 + dZ**2)
N[N == 0] = 1  # Avoid division by zero
dS /= N
dZ /= N

# Plot the vector field
plt.figure(figsize=(10, 6))
plt.quiver(S_values, Z_values, dS, dZ, N)  # Create a 2D field of arrows
plt.xlabel('Number of Susceptible Humans')
plt.ylabel('Number of Zombies')
plt.title('Extended Vector Field of Zombie Apocalypse with Recovery')
plt.grid(True)
plt.xlim([0, 500])
plt.ylim([0, 500])
plt.show()  # Display the plot

