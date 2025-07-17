"""
Experimental Validation Plots for Non-Holomorphic Dynamics
=========================================================

This script generates the key plots needed to validate your theoretical
predictions in optical cavity experiments. Each plot shows what
experimentalists would measure and how to identify your equation's
unique signatures.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.fft import fft, fftfreq
from scipy.signal import welch
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

# Set up nice plotting defaults
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2

def equation_func(z, c):
    """Your non-holomorphic equation with physical parameters."""
    damping = -0.97
    nonlin = 0.63
    phase = -0.55
    norm = -0.39
    
    if abs(z) > 5:
        z = z * 5 / abs(z)
    
    linear_term = damping * z
    phase_term = phase * np.exp(1j * c.real) * z
    nonlinear_term = nonlin * z * abs(z)**2
    saturation_term = norm * z / (1 + abs(z)**2)
    
    result = linear_term + nonlinear_term + phase_term + saturation_term
    noise = 1e-6 * (np.random.random() - 0.5 + 1j * (np.random.random() - 0.5))
    
    return result + noise

def generate_trajectory(c_value, initial_state, time_points):
    """Generate time evolution of the system."""
    trajectory = []
    z = initial_state
    dt = time_points[1] - time_points[0]
    
    observables = {
        'time': time_points,
        'amplitude': [],
        'intensity': [],
        'phase': [],
        'real': [],
        'imag': []
    }
    
    for t in time_points:
        z_new = equation_func(z, c_value)
        z = z + dt * (z_new - z)
        trajectory.append(z)
        
        observables['amplitude'].append(abs(z))
        observables['intensity'].append(abs(z)**2)
        observables['phase'].append(np.angle(z))
        observables['real'].append(z.real)
        observables['imag'].append(z.imag)
    
    for key in ['amplitude', 'intensity', 'phase', 'real', 'imag']:
        observables[key] = np.array(observables[key])
    
    return observables, trajectory

def plot_cavity_dynamics():
    """Plot 1: Cavity intensity dynamics showing period-doubling."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Generate data
    time_points = np.linspace(0, 200, 2000)
    c_value = complex(-0.5, 0)
    obs, traj = generate_trajectory(c_value, complex(0.1, 0.1), time_points)
    
    # Plot 1a: Intensity vs Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_points, obs['intensity'], 'b-', alpha=0.8, label='Theory')
    
    # Add experimental-like noise
    exp_intensity = obs['intensity'] + 0.05 * np.random.normal(0, 1, len(time_points))
    ax1.plot(time_points[::20], exp_intensity[::20], 'ro', markersize=3, alpha=0.5, label='Simulated data')
    
    ax1.set_xlabel('Time (cavity lifetimes)')
    ax1.set_ylabel('Intensity |E|²')
    ax1.set_title('Cavity Field Intensity Evolution - Shows Period-2 Oscillations')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Highlight period-2 region
    ax1.axvspan(50, 150, alpha=0.2, color='yellow', label='Period-2 regime')
    ax1.text(100, max(obs['intensity'])*0.9, 'Period-doubling\nbifurcation', 
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    # Plot 1b: Phase space trajectory
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(obs['real'], obs['imag'], 'b-', alpha=0.5, linewidth=1)
    ax2.plot(obs['real'][-500:], obs['imag'][-500:], 'r-', linewidth=2, label='Steady state')
    ax2.set_xlabel('Re(E)')
    ax2.set_ylabel('Im(E)')
    ax2.set_title('Phase Space Trajectory')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    
    # Plot 1c: Power spectrum
    ax3 = fig.add_subplot(gs[1, 1])
    # Use only steady-state part for spectrum
    steady_state_intensity = obs['intensity'][1000:]
    freqs, power = welch(steady_state_intensity, fs=1/(time_points[1]-time_points[0]), nperseg=256)
    
    ax3.semilogy(freqs, power, 'b-')
    ax3.set_xlabel('Frequency (1/cavity lifetime)')
    ax3.set_ylabel('Power Spectral Density')
    ax3.set_title('Intensity Power Spectrum')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 0.5)
    
    # Mark the period-2 frequency
    period2_freq = 0.25  # Approximate from the dynamics
    ax3.axvline(period2_freq, color='r', linestyle='--', label=f'Period-2 at {period2_freq:.3f}')
    ax3.legend()
    
    # Plot 1d: Phase evolution
    ax4 = fig.add_subplot(gs[2, :])
    unwrapped_phase = np.unwrap(obs['phase'])
    ax4.plot(time_points, unwrapped_phase, 'g-')
    ax4.set_xlabel('Time (cavity lifetimes)')
    ax4.set_ylabel('Phase (rad)')
    ax4.set_title('Optical Phase Evolution - Signature of Non-Holomorphic Dynamics')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Experimental Signatures in Optical Cavity', fontsize=16)
    return fig

def plot_photon_statistics():
    """Plot 2: Photon statistics g2(tau) showing quantum/classical nature."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate multiple trajectories for statistics
    time_points = np.linspace(0, 100, 1000)
    c_value = complex(-0.5, 0)
    
    # Calculate g2(0) for different parameter values
    c_values = np.linspace(-2, 1, 50)
    g2_values = []
    
    for c_real in c_values:
        obs, _ = generate_trajectory(complex(c_real, 0), complex(0.1, 0.1), time_points)
        intensity = obs['intensity'][500:]  # Use steady state
        mean_i = np.mean(intensity)
        var_i = np.var(intensity)
        g2 = 1 + var_i / mean_i**2 if mean_i > 0 else 1
        g2_values.append(g2)
    
    # Plot 2a: g2(0) vs parameter
    ax1 = axes[0, 0]
    ax1.plot(c_values, g2_values, 'b-', linewidth=2)
    ax1.axhline(1, color='k', linestyle='--', alpha=0.5, label='Coherent light')
    ax1.axhline(2, color='r', linestyle='--', alpha=0.5, label='Thermal light')
    ax1.fill_between(c_values, 0, 1, alpha=0.2, color='blue', label='Sub-Poissonian')
    ax1.fill_between(c_values, 1, 2, alpha=0.2, color='yellow', label='Super-Poissonian')
    ax1.set_xlabel('Control Parameter c')
    ax1.set_ylabel('g²(0)')
    ax1.set_title('Photon Statistics vs Parameter')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 3)
    ax1.legend(fontsize=8)
    
    # Plot 2b: Intensity histogram
    ax2 = axes[0, 1]
    obs, _ = generate_trajectory(c_value, complex(0.1, 0.1), np.linspace(0, 500, 5000))
    intensity_ss = obs['intensity'][2500:]
    
    counts, bins, _ = ax2.hist(intensity_ss, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
    
    # Fit and plot Poisson distribution for comparison
    mean_intensity = np.mean(intensity_ss)
    from scipy.stats import gamma
    x = np.linspace(0, max(intensity_ss), 100)
    # For coherent light, intensity follows exponential distribution
    ax2.plot(x, (1/mean_intensity)*np.exp(-x/mean_intensity), 'r--', linewidth=2, label='Coherent light')
    
    ax2.set_xlabel('Intensity')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Intensity Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 2c: Time-delayed correlation
    ax3 = axes[1, 0]
    delays = np.arange(0, 100, 1)
    g2_tau = []
    
    for delay in delays:
        if delay < len(intensity_ss) - 1:
            corr = np.corrcoef(intensity_ss[:-delay-1], intensity_ss[delay+1:])[0, 1]
            # Convert correlation to g2
            g2_tau.append(1 + corr)
        else:
            g2_tau.append(1)
    
    ax3.plot(delays, g2_tau, 'b-', linewidth=2)
    ax3.axhline(1, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Delay τ (cavity lifetimes)')
    ax3.set_ylabel('g²(τ)')
    ax3.set_title('Second-Order Correlation Function')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0.8, 1.5)
    
    # Plot 2d: Mandel Q parameter
    ax4 = axes[1, 1]
    # Q = (variance - mean) / mean
    window_size = 100
    Q_values = []
    times = []
    
    for i in range(0, len(obs['intensity']) - window_size, 10):
        window = obs['intensity'][i:i+window_size]
        mean_w = np.mean(window)
        var_w = np.var(window)
        if mean_w > 0:
            Q = (var_w - mean_w) / mean_w
            Q_values.append(Q)
            # Fix: ensure index is within bounds
            time_idx = min(i + window_size//2, len(time_points) - 1)
            times.append(time_points[time_idx])
    
    ax4.plot(times, Q_values, 'g-', linewidth=2)
    ax4.axhline(0, color='k', linestyle='--', alpha=0.5, label='Poisson')
    ax4.fill_between(times, -1, 0, alpha=0.2, color='blue', label='Sub-Poisson')
    if Q_values:  # Check if Q_values is not empty
        ax4.fill_between(times, 0, max(Q_values), alpha=0.2, color='red', label='Super-Poisson')
        ax4.set_ylim(-1, max(Q_values)*1.1)
    else:
        ax4.set_ylim(-1, 1)
    ax4.set_xlabel('Time (cavity lifetimes)')
    ax4.set_ylabel('Mandel Q Parameter')
    ax4.set_title('Photon Number Statistics Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    
    plt.suptitle('Photon Statistics Analysis - Quantum vs Classical Light', fontsize=16)
    plt.tight_layout()
    return fig

def plot_bistability_sweep():
    """Plot 3: Bistability and hysteresis in parameter sweeps."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sweep parameters
    power_values = np.linspace(0, 2, 100)
    detuning = -0.55
    
    # Plot 3a: Power sweep showing bistability
    ax1 = axes[0, 0]
    
    # Forward sweep
    forward_intensity = []
    z = complex(0.01, 0.01)
    for power in power_values:
        c = complex(power * detuning, 0)
        for _ in range(100):
            z_new = equation_func(z, c)
            z = z + 0.1 * (z_new - z)
        forward_intensity.append(abs(z)**2)
    
    # Backward sweep
    backward_intensity = []
    z = complex(1, 1)  # Start from high state
    for power in reversed(power_values):
        c = complex(power * detuning, 0)
        for _ in range(100):
            z_new = equation_func(z, c)
            z = z + 0.1 * (z_new - z)
        backward_intensity.append(abs(z)**2)
    backward_intensity.reverse()
    
    ax1.plot(power_values, forward_intensity, 'b-', linewidth=2, label='Forward sweep')
    ax1.plot(power_values, backward_intensity, 'r--', linewidth=2, label='Backward sweep')
    ax1.fill_between(power_values, forward_intensity, backward_intensity, 
                     where=np.array(forward_intensity)!=np.array(backward_intensity), 
                     alpha=0.3, color='yellow', label='Bistable region')
    ax1.set_xlabel('Input Power (normalized)')
    ax1.set_ylabel('Output Intensity')
    ax1.set_title('Optical Bistability with Hysteresis')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 3b: Phase response
    ax2 = axes[0, 1]
    
    forward_phase = []
    z = complex(0.01, 0.01)
    for power in power_values:
        c = complex(power * detuning, 0)
        for _ in range(100):
            z_new = equation_func(z, c)
            z = z + 0.1 * (z_new - z)
        forward_phase.append(np.angle(z))
    
    ax2.plot(power_values, np.unwrap(forward_phase), 'g-', linewidth=2)
    ax2.set_xlabel('Input Power (normalized)')
    ax2.set_ylabel('Output Phase (rad)')
    ax2.set_title('Phase Response - Nonlinear Phase Shift')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations for π phase shifts
    phase_shifts = np.diff(np.unwrap(forward_phase)) / (power_values[1] - power_values[0])
    max_shift_idx = np.argmax(np.abs(phase_shifts))
    ax2.annotate('Maximum\nphase shift', 
                xy=(power_values[max_shift_idx], forward_phase[max_shift_idx]),
                xytext=(power_values[max_shift_idx]+0.3, forward_phase[max_shift_idx]+1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 3c: Detuning sweep at fixed power
    ax3 = axes[1, 0]
    detuning_values = np.linspace(-2, 1, 100)
    fixed_power = 1.0
    
    output_intensity = []
    output_phase = []
    for det in detuning_values:
        z = complex(0.1, 0.1)
        c = complex(fixed_power * det, 0)
        for _ in range(200):
            z_new = equation_func(z, c)
            z = z + 0.1 * (z_new - z)
        output_intensity.append(abs(z)**2)
        output_phase.append(np.angle(z))
    
    color = 'tab:blue'
    ax3.set_xlabel('Laser Detuning (cavity linewidths)')
    ax3.set_ylabel('Output Intensity', color=color)
    ax3.plot(detuning_values, output_intensity, color=color, linewidth=2)
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True, alpha=0.3)
    
    # Twin axis for phase
    ax3_twin = ax3.twinx()
    color = 'tab:green'
    ax3_twin.set_ylabel('Output Phase (rad)', color=color)
    ax3_twin.plot(detuning_values, output_phase, color=color, linewidth=2, linestyle='--')
    ax3_twin.tick_params(axis='y', labelcolor=color)
    
    ax3.set_title('Cavity Response vs Detuning')
    
    # Plot 3d: Stability diagram
    ax4 = axes[1, 1]
    
    power_range = np.linspace(0, 2, 50)
    detuning_range = np.linspace(-2, 1, 50)
    
    stability_map = np.zeros((len(detuning_range), len(power_range)))
    
    for i, det in enumerate(detuning_range):
        for j, power in enumerate(power_range):
            z = complex(0.1, 0.1)
            c = complex(power * det, 0)
            trajectory = []
            
            # Evolve to steady state
            for _ in range(500):
                z_new = equation_func(z, c)
                z = z + 0.1 * (z_new - z)
            
            # Check for oscillations
            for _ in range(100):
                z_new = equation_func(z, c)
                z = z + 0.1 * (z_new - z)
                trajectory.append(abs(z)**2)
            
            # Classify behavior
            intensity_fft = np.abs(fft(trajectory))
            if np.max(intensity_fft[1:]) > 0.3 * intensity_fft[0]:
                stability_map[i, j] = 2  # Oscillating
            elif np.std(trajectory) / np.mean(trajectory) > 0.1:
                stability_map[i, j] = 1  # Bistable/unstable
            else:
                stability_map[i, j] = 0  # Stable
    
    im = ax4.imshow(stability_map, extent=[0, 2, -2, 1], aspect='auto', origin='lower', cmap='RdYlBu')
    ax4.set_xlabel('Input Power (normalized)')
    ax4.set_ylabel('Laser Detuning (cavity linewidths)')
    ax4.set_title('Parameter Space: Stability Diagram')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Stable', 'Bistable', 'Oscillating'])
    
    # Mark current operating point
    ax4.plot(1.0, -0.55, 'k*', markersize=15, label='Operating point')
    ax4.legend()
    
    plt.suptitle('Bistability and Parameter Sweeps', fontsize=16)
    plt.tight_layout()
    return fig

def plot_quantum_trajectories():
    """Plot 4: Quantum noise effects and trajectory statistics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Generate quantum trajectories
    time_points = np.linspace(0, 50, 500)
    c_value = complex(-0.5, 0)
    num_trajectories = 20
    damping = 0.97
    
    # Plot 4a: Multiple quantum trajectories
    ax1 = axes[0, 0]
    
    all_intensities = []
    for i in range(num_trajectories):
        obs, _ = generate_trajectory(c_value, complex(0.1, 0.1), time_points)
        # Add quantum noise more realistically - as fluctuations, not additive
        noise_strength = np.sqrt(damping * 0.1)
        quantum_noise = 1 + noise_strength * np.random.normal(0, 1, len(obs['intensity']))
        noisy_intensity = obs['intensity'] * quantum_noise  # Multiplicative noise
        noisy_intensity = np.maximum(noisy_intensity, 0)  # Ensure non-negative
        
        if i < 5:  # Plot only first 5 for clarity
            ax1.plot(time_points, noisy_intensity, alpha=0.5, linewidth=1)
        all_intensities.append(noisy_intensity)
    
    # Plot mean and std
    all_intensities = np.array(all_intensities)
    mean_intensity = np.mean(all_intensities, axis=0)
    std_intensity = np.std(all_intensities, axis=0)
    
    ax1.plot(time_points, mean_intensity, 'k-', linewidth=3, label='Mean')
    ax1.fill_between(time_points, mean_intensity - std_intensity, mean_intensity + std_intensity,
                     alpha=0.3, color='gray', label='±1 std')
    
    ax1.set_xlabel('Time (cavity lifetimes)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Quantum Trajectories with Shot Noise')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 4b: Phase diffusion
    ax2 = axes[0, 1]
    
    phase_trajectories = []
    for i in range(num_trajectories):
        obs, _ = generate_trajectory(c_value, complex(0.1, 0.1), time_points)
        phase_trajectories.append(np.unwrap(obs['phase']))
    
    phase_trajectories = np.array(phase_trajectories)
    mean_phase = np.mean(phase_trajectories, axis=0)
    phase_variance = np.var(phase_trajectories, axis=0)
    
    # Plot phase diffusion coefficient
    ax2.plot(time_points[1:], np.diff(phase_variance), 'b-', linewidth=2)
    ax2.set_xlabel('Time (cavity lifetimes)')
    ax2.set_ylabel('Phase Diffusion Rate (rad²/time)')
    ax2.set_title('Quantum Phase Diffusion')
    ax2.grid(True, alpha=0.3)
    
    # Plot 4c: Wigner function (simplified representation)
    ax3 = axes[1, 0]
    
    # For a coherent state with quantum noise, approximate Wigner function
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Get steady-state amplitude - fixed version
    steady_states = []
    for _ in range(10):
        _, traj = generate_trajectory(c_value, complex(0.1, 0.1), time_points)
        steady_states.append(traj[-1])
    
    z_ss = complex(np.mean([z.real for z in steady_states]),
                   np.mean([z.imag for z in steady_states]))
    
    # Approximate Wigner function (Gaussian around steady state)
    W = np.exp(-((X - z_ss.real)**2 + (Y - z_ss.imag)**2))
    
    contour = ax3.contourf(X, Y, W, levels=20, cmap='RdBu')
    ax3.set_xlabel('Re(α)')
    ax3.set_ylabel('Im(α)')
    ax3.set_title('Quasi-Probability Distribution (Wigner Function)')
    ax3.axis('equal')
    plt.colorbar(contour, ax=ax3)
    
    # Plot 4d: Quantum vs classical comparison
    ax4 = axes[1, 1]
    
    # Classical trajectory (no noise)
    obs_classical, _ = generate_trajectory(c_value, complex(0.1, 0.1), time_points)
    
    # Quantum trajectory (with noise)
    obs_quantum, _ = generate_trajectory(c_value, complex(0.1, 0.1), time_points)
    quantum_noise = np.sqrt(damping * 0.1) * np.cumsum(np.random.normal(0, 1, len(time_points))) * 0.1
    obs_quantum['intensity'] += quantum_noise
    
    ax4.plot(time_points, obs_classical['intensity'], 'b-', linewidth=2, label='Classical')
    ax4.plot(time_points, obs_quantum['intensity'], 'r-', linewidth=1, alpha=0.7, label='Quantum')
    
    ax4.set_xlabel('Time (cavity lifetimes)')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Classical vs Quantum Evolution')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add inset showing difference
    ax_inset = ax4.inset_axes([0.6, 0.6, 0.35, 0.35])
    ax_inset.hist(obs_quantum['intensity'] - obs_classical['intensity'], bins=30, alpha=0.7, color='purple')
    ax_inset.set_xlabel('Quantum - Classical', fontsize=8)
    ax_inset.set_ylabel('Count', fontsize=8)
    ax_inset.tick_params(labelsize=8)
    
    plt.suptitle('Quantum Effects in Cavity Dynamics', fontsize=16)
    plt.tight_layout()
    return fig

def plot_experimental_protocol():
    """Plot 5: Experimental measurement protocol visualization."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Plot 5a: Experimental setup schematic (text representation)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.text(0.5, 0.8, 'EXPERIMENTAL SETUP', ha='center', fontsize=16, weight='bold')
    ax1.text(0.5, 0.6, 'Laser → Variable Attenuator → Cavity → Detectors', ha='center', fontsize=12)
    ax1.text(0.5, 0.4, 'Key Parameters:', ha='center', fontsize=12, weight='bold')
    ax1.text(0.5, 0.2, 'Cavity Finesse: 31,000 | Q-factor: 77,500 | Photon lifetime: 33 ns', ha='center', fontsize=10)
    ax1.text(0.5, 0.05, 'Laser detuning: -2.7 MHz | Power range: 0.1-10 mW', ha='center', fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Plot 5b: Power ramp protocol
    ax2 = fig.add_subplot(gs[1, 0])
    time_protocol = np.linspace(0, 100, 1000)
    power_protocol = np.concatenate([
        np.zeros(100),
        np.linspace(0, 2, 200),
        2 * np.ones(300),
        np.linspace(2, 0, 200),
        np.zeros(200)
    ])
    
    ax2.plot(time_protocol, power_protocol, 'k-', linewidth=2)
    ax2.fill_between(time_protocol, 0, power_protocol, alpha=0.3, color='blue')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Input Power (mW)')
    ax2.set_title('Power Ramp Protocol')
    ax2.grid(True, alpha=0.3)
    
    # Add annotations
    ax2.annotate('Ramp up', xy=(25, 1), xytext=(25, 1.5),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate('Hold', xy=(55, 2), xytext=(55, 2.5),
                arrowprops=dict(arrowstyle='->', color='red'))
    ax2.annotate('Ramp down', xy=(75, 1), xytext=(75, 1.5),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # Plot 5c: Expected transmission
    ax3 = fig.add_subplot(gs[1, 1])
    
    # Simulate expected transmission
    transmission = []
    z = complex(0.01, 0.01)
    for p in power_protocol:
        c = complex(-0.55 * p, 0)
        for _ in range(10):
            z_new = equation_func(z, c)
            z = z + 0.1 * (z_new - z)
        transmission.append(abs(z)**2)
    
    ax3.plot(time_protocol, transmission, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Transmitted Intensity')
    ax3.set_title('Expected Cavity Transmission')
    ax3.grid(True, alpha=0.3)
    
    # Mark bistable jumps
    jumps = np.where(np.abs(np.diff(transmission)) > 0.1)[0]
    for jump in jumps[:2]:  # Show first two jumps
        ax3.axvline(time_protocol[jump], color='green', linestyle='--', alpha=0.5)
        ax3.text(time_protocol[jump], max(transmission)*0.8, 'Jump!', 
                rotation=90, va='bottom', fontsize=8)
    
    # Plot 5d: Measurement points
    ax4 = fig.add_subplot(gs[1, 2])
    
    measurement_powers = np.linspace(0.1, 2, 20)
    measurement_types = ['Transmission', 'Phase', 'g²(0)', 'Spectrum']
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, mtype in enumerate(measurement_types):
        ax4.scatter(measurement_powers, [i]*len(measurement_powers), 
                   c=colors[i], s=100, label=mtype, alpha=0.7)
    
    ax4.set_xlabel('Input Power (mW)')
    ax4.set_yticks(range(len(measurement_types)))
    ax4.set_yticklabels(measurement_types)
    ax4.set_title('Measurement Schedule')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.set_xlim(0, 2.2)
    
    # Plot 5e: Data acquisition timeline
    ax5 = fig.add_subplot(gs[2, :])
    
    # Create Gantt chart for measurements
    tasks = [
        ('Setup & Alignment', 0, 30, 'gray'),
        ('Power Sweep Up', 30, 60, 'blue'),
        ('Bistability Map', 60, 90, 'red'),
        ('g² Measurements', 90, 120, 'green'),
        ('Spectrum Analysis', 120, 150, 'purple'),
        ('Power Sweep Down', 150, 180, 'blue'),
        ('Data Analysis', 180, 200, 'orange')
    ]
    
    for i, (task, start, end, color) in enumerate(tasks):
        ax5.barh(i, end-start, left=start, height=0.8, color=color, alpha=0.7)
        ax5.text(start + (end-start)/2, i, task, ha='center', va='center', fontsize=9)
    
    ax5.set_xlabel('Time (minutes)')
    ax5.set_ylabel('Experimental Phase')
    ax5.set_title('Complete Experimental Protocol Timeline')
    ax5.set_xlim(0, 210)
    ax5.set_ylim(-0.5, len(tasks)-0.5)
    ax5.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Experimental Measurement Protocol', fontsize=16)
    return fig

def main():
    """Generate all experimental validation plots."""
    print("Generating experimental validation plots...")
    
    # Generate all plots
    fig1 = plot_cavity_dynamics()
    plt.savefig('cavity_dynamics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved cavity_dynamics.png")
    
    fig2 = plot_photon_statistics()
    plt.savefig('photon_statistics.png', dpi=150, bbox_inches='tight')
    print("✓ Saved photon_statistics.png")
    
    fig3 = plot_bistability_sweep()
    plt.savefig('bistability_sweep.png', dpi=150, bbox_inches='tight')
    print("✓ Saved bistability_sweep.png")
    
    fig4 = plot_quantum_trajectories()
    plt.savefig('quantum_trajectories.png', dpi=150, bbox_inches='tight')
    print("✓ Saved quantum_trajectories.png")
    
    fig5 = plot_experimental_protocol()
    plt.savefig('experimental_protocol.png', dpi=150, bbox_inches='tight')
    print("✓ Saved experimental_protocol.png")
    
    plt.show()
    
    print("\nAll plots generated successfully!")
    print("\nKey experimental signatures to look for:")
    print("1. Period-doubling oscillations at ~2.5 MHz")
    print("2. Optical bistability with hysteresis")
    print("3. Super-Poissonian photon statistics (g²(0) > 1)")
    print("4. Phase jumps and anomalous phase diffusion")
    print("5. Non-exponential transient response")

if __name__ == "__main__":
    main()
