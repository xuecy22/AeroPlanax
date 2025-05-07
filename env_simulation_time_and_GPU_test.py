import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from functools import partial
from envs.aeroplanax_combat import AeroPlanaxCombatEnv, CombatTaskParams

# 设置XLA内存分配器
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'

# Try to import pynvml for GPU monitoring
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    print("Warning: Unable to import pynvml, GPU monitoring will be disabled")
    HAS_PYNVML = False

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    """Split batch actions into per-agent actions"""
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}

def init_gpu_monitoring():
    """Initialize GPU monitoring"""
    if not HAS_PYNVML:
        return None
    
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            print("No NVIDIA GPU found")
            return None
        
        # Use the first GPU
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return handle
    except:
        print("Failed to initialize GPU monitoring")
        return None

def get_gpu_memory_usage(handle):
    """Get GPU memory usage"""
    if handle is None or not HAS_PYNVML:
        return 0, 0
    
    try:
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / 1024**2, info.total / 1024**2  # Convert to MB
    except:
        return 0, 0

def run_single_benchmark(n_envs, sim_time, dt_phys, gpu_handle):
    """Run benchmark for a specific number of environments"""
    try:
        # Get initial GPU memory usage
        gpu_used_before, _ = get_gpu_memory_usage(gpu_handle)
        
        # Initialize environment
        env_params = CombatTaskParams()
        env = AeroPlanaxCombatEnv(env_params)
        
        # Initialize random number generator
        rng = jax.random.PRNGKey(0)
        
        # Reset environments using vmap
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, n_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)
        
        # Create zero actions for all agents
        action = jnp.zeros((n_envs * env.num_agents, 4))
        
        # Calculate steps
        num_steps = int(sim_time / dt_phys)
        
        # Define step function for scan
        def step_fn(carry, _):
            rng_state, env_state = carry
            rng_step, rng_next = jax.random.split(rng_state)
            step_keys = jax.random.split(rng_step, n_envs)
            unbatched_actions = unbatchify(action, env.agents, n_envs, env.num_agents)
            _, new_state, _, _, _ = jax.vmap(env.step, in_axes=(0, 0, 0))(
                step_keys, env_state, unbatched_actions)
            return (rng_next, new_state), None
        
        # Compile the scan function
        scan_fn = jax.jit(lambda rng, state: jax.lax.scan(
            step_fn, (rng, state), None, length=num_steps
        ))
        
        # Warm up JIT compilation
        _ = scan_fn(rng, env_state)
        
        # Measure actual performance
        start_time = time.time()
        (_, final_state), _ = scan_fn(rng, env_state)
        # Force completion of all computations
        _ = jax.block_until_ready(final_state)
        elapsed_time = time.time() - start_time
        
        # Get GPU memory usage
        gpu_used_after, _ = get_gpu_memory_usage(gpu_handle)
        gpu_memory_used = gpu_used_after - gpu_used_before
        
        # Calculate metrics
        steps_per_sec = n_envs * num_steps / elapsed_time
        real_time_factor = steps_per_sec * dt_phys
        
        print(f"  Time: {elapsed_time:.4f} seconds")
        print(f"  Steps per second: {steps_per_sec:.2f}")
        print(f"  Real-time factor: {real_time_factor:.2f}x")
        print(f"  GPU memory usage: {gpu_memory_used:.2f} MB")
        
        return (n_envs, elapsed_time, steps_per_sec, real_time_factor), (n_envs, gpu_memory_used)
        
    except Exception as e:
        print(f"Error testing {n_envs} environments: {e}")
        return (n_envs, float('inf'), 0.0, 0.0), (n_envs, 0.0)

def plot_results(results, gpu_memory, output_dir="./results"):
    """Plot and save benchmark results"""
    # Extract data, filtering out failed runs (infinite time)
    valid_results = [(n, t, sps, rtf) for n, t, sps, rtf in results if t != float('inf')]
    if not valid_results:
        print("No valid benchmark results to plot")
        return
        
    n_envs, elapsed_times, steps_per_sec, real_time_factors = zip(*valid_results)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set global plot style
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })
    
    # Plot execution time
    plt.figure()
    plt.plot(n_envs, elapsed_times, 'o-', linewidth=2, color='#1f77b4')
    plt.title('Environment Count vs Execution Time')
    plt.xlabel('Number of Environments')
    plt.ylabel('Execution Time (s)')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    # Add x-axis ticks at specific powers of 10
    plt.xticks(n_envs, [f'$10^{int(np.log10(max(n, 1)))}$' for n in n_envs])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'execution_time.png'))
    plt.savefig(os.path.join(output_dir, 'execution_time.pdf'))
    plt.close()
    
    # Plot steps per second
    plt.figure()
    plt.plot(n_envs, steps_per_sec, 'o-', linewidth=2, color='#ff7f0e')
    plt.title('Environment Count vs Steps per Second')
    plt.xlabel('Number of Environments')
    plt.ylabel('Steps per Second')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(n_envs, [f'$10^{int(np.log10(max(n, 1)))}$' for n in n_envs])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steps_per_second.png'))
    plt.savefig(os.path.join(output_dir, 'steps_per_second.pdf'))
    plt.close()
    
    # Plot real-time factor
    plt.figure()
    plt.plot(n_envs, real_time_factors, 'o-', linewidth=2, color='#2ca02c')
    plt.title('Environment Count vs Real-time Factor')
    plt.xlabel('Number of Environments')
    plt.ylabel('Real-time Factor (x)')
    plt.xscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(n_envs, [f'$10^{int(np.log10(max(n, 1)))}$' for n in n_envs])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'real_time_factor.png'))
    plt.savefig(os.path.join(output_dir, 'real_time_factor.pdf'))
    plt.close()
    
    # Plot GPU memory usage if available
    if gpu_memory:
        valid_gpu_memory = [(n, mem) for (n, mem), (_, t, _, _) in zip(gpu_memory, results) if t != float('inf')]
        if valid_gpu_memory:
            n_envs_gpu, memory_used = zip(*valid_gpu_memory)
            plt.figure()
            plt.plot(n_envs_gpu, memory_used, 'o-', linewidth=2, color='#d62728')
            plt.title('Environment Count vs GPU Memory Usage')
            plt.xlabel('Number of Environments')
            plt.ylabel('GPU Memory Usage (MB)')
            plt.xscale('log')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(n_envs_gpu, [f'$10^{int(np.log10(max(n, 1)))}$' for n in n_envs_gpu])
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gpu_memory.png'))
            plt.savefig(os.path.join(output_dir, 'gpu_memory.pdf'))
            plt.close()
    
    # Create summary report
    with open(os.path.join(output_dir, 'benchmark_report.txt'), 'w') as f:
        f.write("AeroPlanax Environment Performance Benchmark Report\n")
        f.write("=================================================\n\n")
        
        f.write("Test Configuration:\n")
        f.write(f"- Environment counts: {', '.join(map(str, n_envs))}\n")
        f.write(f"- JAX devices: {jax.devices()}\n\n")
        
        f.write("Performance Results:\n")
        for i, n in enumerate(n_envs):
            if i < len(elapsed_times):
                f.write(f"Environment count: {n} (10^{int(np.log10(max(n, 1)))})\n")
                f.write(f"  Execution time: {elapsed_times[i]:.4f} s\n")
                f.write(f"  Steps per second: {steps_per_sec[i]:.2f}\n")
                f.write(f"  Real-time factor: {real_time_factors[i]:.2f}x\n")
                if i < len(gpu_memory) and gpu_memory[i][1] > 0:
                    f.write(f"  GPU memory usage: {gpu_memory[i][1]:.2f} MB\n")
                f.write("\n")
    
    print(f"Benchmark results saved to {output_dir}")

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AeroPlanax Environment Benchmark')
    parser.add_argument('--sim_time', type=float, default=10.0, help='Simulation time per environment (seconds)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    args = parser.parse_args()
    
    # Initialize a sample environment to get its parameters
    env_params = CombatTaskParams()
    sim_freq = env_params.sim_freq  # Should be 50
    
    # Calculate the time step for each physical simulation step
    dt_phys = 1.0 / sim_freq  # 1/50 = 0.02 seconds
    
    # Calculate number of physical simulation steps
    num_steps = int(args.sim_time * sim_freq)  # 10 seconds * 50 steps/second = 500 steps
    
    print(f"Environment simulation frequency: {sim_freq} Hz")
    print(f"Physical simulation step time: {dt_phys} seconds")
    print(f"Testing {num_steps} physical simulation steps over {args.sim_time} simulated seconds")
    
    # Generate list of environment counts: 10^0, 10^1, 10^2, 10^3, 10^4, 10^5, 10^6
    powers = range(0, 7)  # 0 to 6 inclusive
    n_list = [10**power for power in powers]
    
    print(f"AeroPlanax Benchmark: 10^0 → 10^6 environments, {args.sim_time}s each")
    
    # Initialize GPU monitoring
    gpu_handle = init_gpu_monitoring()
    
    # Run benchmarks
    results = []
    gpu_memory = []
    
    for n_envs in n_list:
        print(f"Testing {n_envs} environments...")
        result, gpu_mem = run_single_benchmark(n_envs, args.sim_time, dt_phys, gpu_handle)
        results.append(result)
        gpu_memory.append(gpu_mem)
        
        # If this test failed with too many environments, skip larger ones
        if result[1] == float('inf') and n_envs >= 100000:
            print(f"Skipping larger environment counts due to failure")
            break
    
    # Clean up GPU monitoring
    if HAS_PYNVML:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    
    # Plot and save results
    plot_results(results, gpu_memory, args.output_dir)
    
    print(f"Benchmark completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()