import sys
# import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Ensure the script can import the 'common' module from the parent directory
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from common.recording_data import load_recording_data
from definitions import OutputSuffixes

def plot_joint_trajectories(file_path):
    print(f"[Plotter] Loading MMRT data from: {file_path}")
    
    try:
        mmrt = load_recording_data(file_path)
    except Exception as e:
        print(f"[Plotter Error] Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)
        
    fps = mmrt.fps
    dt = 1.0 / fps if fps > 0 else 0.01
    
    for label, pos_data in mmrt.labeled_data.items():
        # Calculate velocity and acceleration
        vel_data = np.gradient(pos_data, dt)
        acc_data = np.gradient(vel_data, dt)
        time_axis = np.arange(len(pos_data)) * dt
        
        # Create a separated window for each joint
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        fig.canvas.manager.set_window_title(f"Joint Evaluation: {label}")
        
        # Position
        axs[0].plot(time_axis, pos_data, 'b-', label='Position')
        axs[0].set_ylabel('Pos (rad or m)')
        axs[0].legend(loc='upper right')
        axs[0].grid(True)
        
        # Velocity
        axs[1].plot(time_axis, vel_data, 'g-', label='Velocity')
        axs[1].set_ylabel('Vel (rad/s or m/s)')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)
        
        # Acceleration
        axs[2].plot(time_axis, acc_data, 'r-', label='Acceleration')
        axs[2].set_ylabel('Acc (rad/s² or m/s²)')
        axs[2].set_xlabel('Time (s)')
        axs[2].legend(loc='upper right')
        axs[2].grid(True)
        
        plt.tight_layout()
        
    print(f"[Plotter] Displaying {len(mmrt.labeled_data)} windows. Close them to exit.")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_joint_trajectories(sys.argv[1])
    else:
        print(
            f"Error: Please provide the path to the {OutputSuffixes.YAML_RECORDING} record file.",
            file=sys.stderr,
        )
        sys.exit(1)
