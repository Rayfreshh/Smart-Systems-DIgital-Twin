import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Digital_twin import DigitalTwin
from bayes_opt import BayesianOptimization

# Initialize the digital twin object
digital_twin = DigitalTwin()

# Load the dataset
csv_file_path = 'filtered_data.csv'
df = pd.read_csv(csv_file_path)
df_time = df['time_seconds']
df_theta = df['theta_filtered']

# Convert theta from degrees to radians
#df_theta = np.radians(df_theta)

def find_initial_state(df_theta, df_time):
    # Find the initial conditions of theta and theta_dot in the dataset by calculating the difference between each data point
    #  Initial theta value 
    theta = df_theta.iloc[0]
   
    # Choose a window size or 'polyorder' appropriate for your data
    theta_dot = (df_theta.iloc[1] - df_theta.iloc[0]) / (df_time.iloc[1] - df_time.iloc[0])
    return theta, theta_dot

# Uncomment the following lines to define and initialize theta and theta_dot
theta, theta_dot = find_initial_state(df_theta, df_time)

print(theta_dot)


# Simulation parameters for the digital twin model
delta_t = (df_time.iloc[-1] - df_time.iloc[0]) / len(df_time)
print(delta_t)

sim_time = (12,25)


# Parameter ranges for simulation
c_air_range = np.linspace(0, 0.4, 15)
c_c_range = np.linspace(0, 0.5, 15)
g = 9.81
l_range = np.linspace(0.2, 0.4, 10)

def simulate_potential_model(theta, theta_dot, c_air, c_c, g, l, num_steps):
    # Simulate the pendulum motion based on the digital twin model
    digital_twin.c_air = c_air
    digital_twin.c_c = c_c
    digital_twin.g = g
    digital_twin.l = l
    sim_measurements = []
    for _ in range(num_steps):
        theta_double_dot = digital_twin.get_theta_double_dot(theta, theta_dot)
        theta += theta_dot * delta_t
        theta_dot += theta_double_dot * delta_t
        sim_measurements.append(theta)
    return sim_measurements

# Initialize search for best parameters
best_params = None
lowest_error = float('inf')

# Perform a grid search over the parameter space
for c_air in c_air_range:
    for c_c in c_c_range:
        for l in l_range:
            simulated_theta = simulate_potential_model(theta, theta_dot, c_air, c_c, g, l, len(df_theta))
            error = np.sqrt(np.mean((np.array(df_theta) - np.array(simulated_theta))**2))
            if error < lowest_error:
                lowest_error = error
                best_params = (c_air, c_c, l)
                print(f"{error} found a better error with parameters: c_air={c_air}, c_c={c_c}, l={l}")

# Display the best parameters and the lowest error
print("GRID Best Parameters:", best_params)
print("GRID Lowest Error:", lowest_error)

# Define the function to optimize the parameters using Bayesian optimization
def optimize_parameters(c_air, c_c, l):
    simulated_theta = simulate_potential_model(theta, theta_dot, c_air, c_c, g, l, len(df_theta)) # Simulate the model
    error = np.sqrt(np.mean((np.array(df_theta) - np.array(simulated_theta))**2)) # Calculate the error
    return -error  # Maximizing the negative error

# Define parameter bounds for Bayesian optimization
pbounds = {'c_air': (0, 0.4), 'c_c': (0, 0.5), 'l': (0.2, 0.4)} # Parameter bounds for Bayesian optimization

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(f=optimize_parameters, pbounds=pbounds, random_state=1) 

# Begin Optimization Process (This is where the Bayesian optimization internally iterates)
print("Starting Bayesian Optimization")
optimizer.maximize(init_points=15, n_iter=30)    # Perform 15 random initializations, then perform 30 Bayesian iterations

# Retrieve Best Parameters and Lowest Error from Bayesian Optimization
bayesian_best_params = optimizer.max['params'] # Best parameters found by Bayesian optimization
bayesian_lowest_error = -optimizer.max['target']  # Note: Optimizer maximizes the function, so we take negative of max 'target'

print("Bayesian Best Parameters:", bayesian_best_params)
print("Bayesian Lowest Error:", bayesian_lowest_error)

# Plot the measured vs. simulated data using both grid search and Bayesian optimization results
grid_best_c_air, grid_best_c_c, grid_best_l = best_params
bayesian_best_c_air, bayesian_best_c_c, bayesian_best_l = bayesian_best_params['c_air'], bayesian_best_params['c_c'], bayesian_best_params['l']

# Simulate both grid and Bayesian results
simulated_theta_grid = simulate_potential_model(theta, theta_dot, grid_best_c_air, grid_best_c_c, g, grid_best_l, len(df_theta))
simulated_theta_bayesian = simulate_potential_model(theta, theta_dot, bayesian_best_c_air, bayesian_best_c_c, g, bayesian_best_l, len(df_theta))

#plot the measured , grid and bayesian simulated data
plt.figure()
plt.plot(df_time, df_theta, label='Filtered Data')
plt.plot(df_time, simulated_theta_grid, label='Grid Search Simulated Data')
plt.plot(df_time, simulated_theta_bayesian, label='Bayesian Optimization Simulated Data' , linestyle='dashed')
plt.xlabel('Time (s)')
plt.ylabel('Theta (rad)')
plt.legend()
plt.show()

