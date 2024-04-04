import time
from Digital_twin import DigitalTwin
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
from scipy.signal import butter, filtfilt

# Before starting run pip install -r requirements.txt

digital_twin = DigitalTwin()

def butterworth_filter(data, cutoff_freq, fs, order=5):
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Butterworth Filter setup
fs = 100.0  # Sample rate, Hz (adjust based on your data)
cutoff = 1.5  # Desired cutoff frequency of the filter, Hz

# Load the recording
digital_twin.load_recording("test_data")

# Apply the Butterworth filter to the theta column of the dataframe
digital_twin.df['filtered_theta'] = butterworth_filter(digital_twin.df['theta'], cutoff, fs)

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance  # Process variance
        self.measurement_variance = measurement_variance  # Measurement variance
        self.estimated_measurement_variance = estimated_measurement_variance  # Estimated measurement variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        # Prediction update
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Measurement update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate
    

theta_original = []
theta_filtered_median = []
theta_filtered_ema = []
theta_filtered_kalman = []

# Exponential Moving Average (EMA) setup
alpha = 0.3  # Smoothing factor for EMA
ema = 1  # Initial EMA value

# Median Filter setup
kernel_size = 31  # Kernel size for the median filter (must be odd)
buffer = np.zeros(kernel_size)  # Initialize buffer with zeros

# Initialize the Kalman Filter for theta (Adjust the variances as needed)
kalman_filter_theta = KalmanFilter(process_variance=1e-5, measurement_variance=1e-5, estimated_measurement_variance=1e-5)
        
if __name__=='__main__':
        digital_twin.load_recording("test_data")
        
        for i in range(len(digital_twin.df)-1):
            #replay the recording:
            sim_time, theta, x_pivot = digital_twin.recorded_step(i)
    	    #Kalman filter
            filtered_kalman = kalman_filter_theta.update(theta)
            # EMA
            filtered_ema = alpha * theta + (1 - alpha) * ema
            # Update the median filter buffer with the new theta value
            buffer = np.roll(buffer, -1)  # Shift everything to the left
            buffer[-1] = theta  # Insert the new theta at the end of the buffer
            # Apply the median filter to the buffer
            if i >= kernel_size - 1:  # Wait until the buffer is fully populated
                filtered_med = medfilt(buffer, kernel_size=kernel_size)[kernel_size//2]
            else:
                filtered_med = theta  # Use the original theta until the buffer is full

            # Store the results for plotting
            theta_original.append(theta)
            theta_filtered_median.append(filtered_med)
            theta_filtered_ema.append(filtered_ema)
            theta_filtered_kalman.append(filtered_kalman)
            #Uncoment this section and change theta to a filtered theta
            #digital_twin.render(theta, x_pivot)
            #delay = (sim_time - pref_sim_time)/1000
            #time.sleep(delay)
            #pref_sim_time = sim_time
            

        plt.figure(figsize=(12, 6))
        plt.plot(theta_original, label='Original Theta')
        plt.plot(theta_filtered_median, label='Median Filtered Theta', linestyle='--')
        plt.plot(theta_filtered_kalman, label='Kalman Filtered Theta', linestyle='-')
        plt.plot(theta_filtered_ema, label='EMA Filtered Theta', linestyle=':')
        plt.title('Original vs. Filtered Theta')
        plt.xlabel('Sample')
        plt.ylabel('Theta Value')
        plt.legend()
        plt.show()