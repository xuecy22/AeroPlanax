import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Define smoothing function
def smoother(x, a=0.9, w=1, mode="window"):
    if mode == "window":
        y = []
        for i in range(len(x)):
            y.append(np.mean(x[max(i - w, 0):i + 1]))
    elif mode == "moving":
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((1 - a) * x[i] + a * y[i - 1])
    else:
        raise NotImplementedError
    return y

# Process single dataset and apply smoothing
def process_single_run_data(data, window_size=80, scale=1.0):
    # Apply smoothing
    smoothed_data = smoother(np.asarray(data), w=window_size, mode="window")
    x = np.arange(0, len(smoothed_data)) * scale
    return x, smoothed_data

# Read data from tensorboard logs
def read_tensorboard_data(log_dir, tag):
    try:
        ea = event_accumulator.EventAccumulator(
            log_dir,
            size_guidance={
                event_accumulator.SCALARS: 0,  # Load all scalars
            }
        )
        ea.Reload()
        
        if tag not in ea.scalars.Keys():
            print(f"Tag {tag} not found in {log_dir}")
            return None
        
        # Get data and convert to DataFrame
        events = ea.Scalars(tag)
        data = pd.DataFrame([(e.step, e.value) for e in events], 
                           columns=['step', 'value'])
        return data
    except Exception as e:
        print(f"Error processing {log_dir}: {e}")
        return None

# Process task data
def process_task_data(log_dirs, metric, window_size=80, steps_per_point=300*3000/1e6):
    # Store processed data from all seeds
    all_seed_curves = []
    min_length = float('inf')
    
    # Read and process data for each seed
    for log_dir in log_dirs:
        data = read_tensorboard_data(log_dir, metric)
        if data is None or data.empty:
            print(f"Warning: Could not read data from {log_dir}")
            continue
        
        # Apply smoothing
        _, smoothed_data = process_single_run_data(data['value'].values, window_size=window_size)
        all_seed_curves.append(smoothed_data)
        
        # Track minimum length
        min_length = min(min_length, len(smoothed_data))
    
    # If no valid data, return None
    if not all_seed_curves:
        return None, None, None, 0
        
    # Truncate all curves to the same length
    all_seed_curves = [curve[:min_length] for curve in all_seed_curves]
    
    # Convert to numpy array and calculate mean and standard deviation
    all_seed_curves = np.array(all_seed_curves)
    mean_curve = np.mean(all_seed_curves, axis=0)
    std_curve = np.std(all_seed_curves, axis=0)
    
    # Create x-axis (using environment steps as units, in millions)
    x = np.arange(0, min_length) * steps_per_point
    
    return x, mean_curve, std_curve, min_length

# Main function
def plot_combined_results():
    # Define log directory paths for both tasks
    heading_log_dirs = [
        # Heading task(seed:0/10/42)
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/heading/heading policy(seed 0)/logs_seed_0",
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/heading/heading policy(seed 10)/logs_seed_10",
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/heading/heading policy(seed 42)/logs_seed_42"
    ]

    ultimate_log_dirs = [
        # Ultimate Goal task(seed:0/10/42)
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/ultimate goal/ultimate_goal(max_15.0)_seed_0/logs",
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/ultimate goal/ultimate_goal(max_15.0)_seed_10/logs",
        "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/ultimate goal/ultimate_goal(max_15.0)_seed_42/logs"
    ]
    
    # Only plot episode reward metric
    metric = "eval/episodic_return"
    
    # Set plotting style
    sns.set_theme(
        style="darkgrid",
        font_scale=1.2,
        rc={"figure.figsize": (10, 6)}
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    colors = sns.color_palette("Set1", 2)  # Use 2 colors for 2 tasks
    
    # Process Heading task data
    print("\nProcessing Heading task data")
    x_heading, mean_heading, std_heading, len_heading = process_task_data(heading_log_dirs, metric)
    
    # Process Ultimate Goal task data
    print("\nProcessing Ultimate Goal task data")
    x_ultimate, mean_ultimate, std_ultimate, len_ultimate = process_task_data(ultimate_log_dirs, metric)
    
    # Find the minimum length between the two datasets
    if x_heading is not None and x_ultimate is not None:
        min_x_len = min(len(x_heading), len(x_ultimate))
        
        # Truncate data to the minimum length
        x_heading = x_heading[:min_x_len]
        mean_heading = mean_heading[:min_x_len]
        std_heading = std_heading[:min_x_len]
        
        x_ultimate = x_ultimate[:min_x_len]
        mean_ultimate = mean_ultimate[:min_x_len]
        std_ultimate = std_ultimate[:min_x_len]
    
    # Plot both tasks if data available
    if x_heading is not None and mean_heading is not None:
        # Plot Heading task mean curve
        ax.plot(x_heading, mean_heading, color=colors[0], label="Heading Task", linewidth=2)
        # Add standard deviation shadow
        ax.fill_between(x_heading, mean_heading - std_heading, mean_heading + std_heading, color=colors[0], alpha=0.2)
        print(f"Heading task final value: {mean_heading[-1]:.2f} ± {std_heading[-1]:.2f}")
    
    if x_ultimate is not None and mean_ultimate is not None:
        # Plot Ultimate Goal task mean curve
        ax.plot(x_ultimate, mean_ultimate, color=colors[1], label="Ultimate Goal Task", linewidth=2)
        # Add standard deviation shadow
        ax.fill_between(x_ultimate, mean_ultimate - std_ultimate, mean_ultimate + std_ultimate, color=colors[1], alpha=0.2)
        print(f"Ultimate Goal task final value: {mean_ultimate[-1]:.2f} ± {std_ultimate[-1]:.2f}")
    
    # Set chart title and labels
    ax.set_title("Heading/Ultimate Goal Training Performance Comparison", fontsize=14)
    ax.set_xlabel("Million Environment Steps", fontsize=12)
    ax.set_ylabel("Average Reward", fontsize=12)
    ax.legend(loc="lower right", fontsize=12)  # Position legend at lower right
    
    # Show grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    plt.tight_layout()
    output_dir = "/home/dqy/aeroplanax/AeroPlanax_heading/plot/plot_result/new/heading/heading_and_ultimate_goal/"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}ultimate goal comparison.png", bbox_inches="tight")
    plt.savefig(f"{output_dir}ultimate goal comparison.pdf", bbox_inches="tight")
    print(f"\nChart saved as {output_dir}ultimate goal comparison.png and {output_dir}ultimate goal comparison.pdf")
    plt.show()

# Execute main function
if __name__ == "__main__":
    plot_combined_results()