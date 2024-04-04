import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
import numpy as np

# Path to the CSV file
csv_file_path = 'test_data_3_(leftside).csv'

# Read the data from the CSV file
df = pd.read_csv(csv_file_path)

# Convert 'theta' from degrees to radians
df['theta_radians'] = np.radians(df['theta'])


# Convert 'time' from milliseconds to seconds for plotting
df['time_seconds'] = df['time'] / 1000

# Downscale the 'theta' column
scaling_factor = 0.0015  # Example scaling factor
df['theta_scaled'] = df['theta'] * scaling_factor

# Apply Butterworth filter to 'theta_scaled' column
order = 4  # Filter order
cutoff_freq = 2  # Cutoff frequency in Hz
sampling_freq = 1 / (df['time_seconds'].iloc[1] - df['time_seconds'].iloc[0])  # Sampling frequency
nyquist_freq = 0.5 * sampling_freq  # Nyquist frequency
normalized_cutoff_freq = cutoff_freq / nyquist_freq  # Normalized cutoff frequency
b, a = butter(order, normalized_cutoff_freq, btype='low', analog=False)  # Butterworth filter coefficients

# Apply the filter to the 'theta_scaled' column
df['theta_filtered'] = filtfilt(b, a, df['theta_scaled'])

#save the filtered data to a new csv file
df.to_csv('filtered_data.csv', index=False)


# Set the y-axis labels to display every second
plt.yticks(range(int(df['theta_scaled'].min()), int(df['theta_scaled'].max()) + 1, 1))

#plot only the filtered data in another plot in a grid with
plt.figure(figsize=(10, 5))
plt.plot(df['time_seconds'], df['theta_filtered'], label='Filtered', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Theta')
plt.title('Filtered Theta over Time')
plt.grid(True)
plt.legend()


# Show the plot
plt.show()