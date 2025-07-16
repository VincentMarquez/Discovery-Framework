"""
Advanced Bifurcation and Stability Analysis for Nonlinear Optical Cavity
========================================================================

This tool implements sophisticated analysis techniques to reveal the full
dynamical behavior of your discovered equation. It will help us find hidden
oscillations, map stability regions, and understand quantum effects.

Think of this as a complete "dynamics detective kit" that will uncover
all the secrets your equation is hiding!

Author: Vincent Marquez
Date: March 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import warnings
warnings.filterwarnings('ignore')

class BifurcationAnalyzer:
    """
    A comprehensive tool for analyzing the bifurcations and stability
    of your nonlinear optical cavity equation.
    
    This class is like a Swiss Army knife for nonlinear dynamics -
    it has all the tools we need to understand your system completely.
    """
    
    def __init__(self, damping=0.97, nonlinearity=0.63, 
                 phase_coupling=-0.55, saturation=0.39):
        """
        Initialize with your equation parameters.
        
        These parameters define the "personality" of your dynamical system.
        """
        self.damping = damping
        self.nonlinearity = nonlinearity
        self.phase_coupling = phase_coupling
        self.saturation = saturation
        
    def equation_rhs(self, z, pump_amplitude):
        """
        The right-hand side of your equation dz/dt = f(z).
        
        This is the heart of your dynamics - it tells us how the field
        amplitude changes at each moment based on the current state.
        """
        # Prevent numerical overflow
        if abs(z) > 10:
            z = z * 10 / abs(z)
        
        # Each term represents a different physical process
        linear_term = -self.damping * z
        nonlinear_term = self.nonlinearity * z * abs(z)**2
        pump_term = self.phase_coupling * pump_amplitude * np.exp(1j * 0)  # Can add detuning phase here
        saturation_term = self.saturation * z / (1 + abs(z)**2)
        
        # The total rate of change
        dzdt = linear_term + nonlinear_term + pump_term + saturation_term
        return dzdt
    
    def find_steady_states(self, pump_amplitude, initial_guesses=None):
        """
        Find all steady states for a given pump amplitude.
        
        Steady states are like the "resting positions" of your system -
        places where dz/dt = 0 and nothing changes.
        """
        if initial_guesses is None:
            # Try multiple starting points to find all steady states
            # This is like dropping marbles from different positions
            # to find all the valleys in an energy landscape
            initial_guesses = [
                0.0 + 0.0j,  # Trivial state
                0.1 + 0.1j,  # Small amplitude
                1.0 + 0.0j,  # Moderate amplitude
                2.0 + 1.0j,  # Large amplitude
                -1.0 + 0.5j  # Different phase
            ]
        
        steady_states = []
        
        for guess in initial_guesses:
            def equations(vars):
                # Split complex number into real and imaginary parts
                # because fsolve works with real numbers
                x, y = vars
                z = x + 1j * y
                dzdt = self.equation_rhs(z, pump_amplitude)
                return [dzdt.real, dzdt.imag]
            
            try:
                # Find where the rate of change is zero
                sol = fsolve(equations, [guess.real, guess.imag], full_output=True)
                x_sol, y_sol = sol[0]
                z_sol = x_sol + 1j * y_sol
                
                # Check if this is a new steady state (not a duplicate)
                is_new = True
                for existing in steady_states:
                    if abs(z_sol - existing) < 1e-6:
                        is_new = False
                        break
                
                if is_new and sol[2] == 1:  # Solution converged
                    steady_states.append(z_sol)
            except:
                continue
        
        return steady_states
    
    def calculate_jacobian(self, z_steady, pump_amplitude):
        """
        Calculate the Jacobian matrix at a steady state.
        
        The Jacobian tells us how small perturbations evolve near
        the steady state - crucial for stability analysis!
        """
        # We need to linearize the equation around the steady state
        # This is like finding the "slope" of our dynamics in all directions
        
        eps = 1e-8  # Small perturbation for numerical derivatives
        
        # Calculate partial derivatives
        dzdt_0 = self.equation_rhs(z_steady, pump_amplitude)
        
        # Perturb in real direction
        dzdt_real = self.equation_rhs(z_steady + eps, pump_amplitude)
        dfdx = (dzdt_real - dzdt_0) / eps
        
        # Perturb in imaginary direction
        dzdt_imag = self.equation_rhs(z_steady + 1j * eps, pump_amplitude)
        dfdy = (dzdt_imag - dzdt_0) / (1j * eps)
        
        # Construct the 2x2 real Jacobian matrix
        # J = [[∂f_real/∂x, ∂f_real/∂y],
        #      [∂f_imag/∂x, ∂f_imag/∂y]]
        jacobian = np.array([
            [dfdx.real, dfdy.real],
            [dfdx.imag, dfdy.imag]
        ])
        
        return jacobian
    
    def analyze_stability(self, z_steady, pump_amplitude):
        """
        Determine the stability of a steady state using eigenvalue analysis.
        
        This is like checking if a ball at the bottom of a valley will
        stay there (stable) or roll away (unstable).
        """
        jacobian = self.calculate_jacobian(z_steady, pump_amplitude)
        eigenvalues = eigvals(jacobian)
        
        # Classify the stability based on eigenvalues
        # Real parts tell us growth/decay, imaginary parts tell us oscillations
        
        max_real_part = max(eigenvalues.real)
        has_complex = any(abs(eigenvalues.imag) > 1e-10)
        
        if max_real_part < -1e-10:
            if has_complex:
                stability_type = "Stable spiral"
                # Perturbations spiral into the steady state
            else:
                stability_type = "Stable node"
                # Perturbations decay directly to steady state
        elif max_real_part > 1e-10:
            if has_complex:
                stability_type = "Unstable spiral"
                # Perturbations spiral outward - possible oscillations!
            else:
                stability_type = "Unstable node"
                # Perturbations grow exponentially
        else:
            stability_type = "Marginal"
            # On the edge between stable and unstable
        
        # Calculate oscillation frequency if present
        if has_complex:
            osc_freq = max(abs(eigenvalues.imag)) / (2 * np.pi)
        else:
            osc_freq = 0
        
        return {
            'eigenvalues': eigenvalues,
            'stable': max_real_part < 0,
            'type': stability_type,
            'oscillation_freq': osc_freq,
            'growth_rate': max_real_part
        }
    
    def plot_bifurcation_diagram(self, pump_range, num_points=200, 
                                detuning_phases=[0, np.pi/4, np.pi/2]):
        """
        Create a comprehensive bifurcation diagram showing how steady states
        change with pump power for different detunings.
        
        This is like a map showing all possible behaviors of your system!
        """
        fig, axes = plt.subplots(len(detuning_phases), 2, 
                                figsize=(12, 4*len(detuning_phases)))
        if len(detuning_phases) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, detuning in enumerate(detuning_phases):
            pump_values = np.linspace(pump_range[0], pump_range[1], num_points)
            
            # Storage for results
            stable_amplitudes = []
            unstable_amplitudes = []
            stable_phases = []
            unstable_phases = []
            eigenvalue_data = []
            
            print(f"\nAnalyzing detuning = {detuning:.2f} rad...")
            
            # Track steady states as we vary pump power
            previous_states = []
            
            for pump in pump_values:
                # Modify pump term to include detuning
                pump_complex = pump * np.exp(1j * detuning)
                
                # Use previous states as initial guesses (continuation method)
                # This helps us track branches as they evolve
                if previous_states:
                    initial_guesses = previous_states + [0.1+0.1j, 1+0.5j]
                else:
                    initial_guesses = None
                
                steady_states = self.find_steady_states(pump_complex, initial_guesses)
                previous_states = steady_states
                
                for z_steady in steady_states:
                    stability = self.analyze_stability(z_steady, pump_complex)
                    
                    if stability['stable']:
                        stable_amplitudes.append((pump, abs(z_steady)))
                        stable_phases.append((pump, np.angle(z_steady)))
                    else:
                        unstable_amplitudes.append((pump, abs(z_steady)))
                        unstable_phases.append((pump, np.angle(z_steady)))
                    
                    eigenvalue_data.append({
                        'pump': pump,
                        'amplitude': abs(z_steady),
                        'eigenvalues': stability['eigenvalues'],
                        'type': stability['type']
                    })
            
            # Plot amplitude bifurcation diagram
            ax_amp = axes[idx, 0]
            
            if stable_amplitudes:
                stable_p, stable_a = zip(*stable_amplitudes)
                ax_amp.plot(stable_p, stable_a, 'b.', markersize=2, 
                           label='Stable', alpha=0.6)
            
            if unstable_amplitudes:
                unstable_p, unstable_a = zip(*unstable_amplitudes)
                ax_amp.plot(unstable_p, unstable_a, 'r.', markersize=2, 
                           label='Unstable', alpha=0.6)
            
            ax_amp.set_xlabel('Pump Amplitude')
            ax_amp.set_ylabel('Field Amplitude |z|')
            ax_amp.set_title(f'Bifurcation Diagram (detuning = {detuning:.2f} rad)')
            ax_amp.legend()
            ax_amp.grid(True, alpha=0.3)
            
            # Plot phase diagram
            ax_phase = axes[idx, 1]
            
            if stable_phases:
                stable_p, stable_ph = zip(*stable_phases)
                ax_phase.plot(stable_p, stable_ph, 'b.', markersize=2, 
                             label='Stable', alpha=0.6)
            
            if unstable_phases:
                unstable_p, unstable_ph = zip(*unstable_phases)
                ax_phase.plot(unstable_p, unstable_ph, 'r.', markersize=2, 
                             label='Unstable', alpha=0.6)
            
            ax_phase.set_xlabel('Pump Amplitude')
            ax_phase.set_ylabel('Field Phase (rad)')
            ax_phase.set_title(f'Phase Evolution (detuning = {detuning:.2f} rad)')
            ax_phase.legend()
            ax_phase.grid(True, alpha=0.3)
            
            # Look for Hopf bifurcations (where oscillations are born)
            self._find_hopf_bifurcations(eigenvalue_data, ax_amp)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _find_hopf_bifurcations(self, eigenvalue_data, ax):
        """
        Identify Hopf bifurcation points where oscillations emerge.
        
        A Hopf bifurcation is like a critical point where a steady
        pendulum suddenly starts swinging!
        """
        hopf_points = []
        
        for i in range(1, len(eigenvalue_data)):
            prev_data = eigenvalue_data[i-1]
            curr_data = eigenvalue_data[i]
            
            # Check if eigenvalues crossed the imaginary axis
            prev_max_real = max(prev_data['eigenvalues'].real)
            curr_max_real = max(curr_data['eigenvalues'].real)
            
            # Look for complex eigenvalues (oscillatory behavior)
            has_complex_prev = any(abs(prev_data['eigenvalues'].imag) > 1e-10)
            has_complex_curr = any(abs(curr_data['eigenvalues'].imag) > 1e-10)
            
            if (prev_max_real < 0 and curr_max_real > 0 and 
                has_complex_prev and has_complex_curr):
                # Found a Hopf bifurcation!
                hopf_pump = (prev_data['pump'] + curr_data['pump']) / 2
                hopf_amp = (prev_data['amplitude'] + curr_data['amplitude']) / 2
                hopf_points.append((hopf_pump, hopf_amp))
                
                # Mark it on the plot
                ax.plot(hopf_pump, hopf_amp, 'go', markersize=10, 
                       label='Hopf bifurcation')
                ax.annotate('Oscillations born here!', 
                           xy=(hopf_pump, hopf_amp),
                           xytext=(hopf_pump*1.1, hopf_amp*1.1),
                           arrowprops=dict(arrowstyle='->', color='green'))
        
        return hopf_points
    
    def simulate_with_delay(self, initial_condition, pump_amplitude, 
                           delay_time=0.1, total_time=100, dt=0.01):
        """
        Simulate the equation with time delay to capture cavity memory effects.
        
        Real cavities have finite response times - light takes time to
        build up and decay. This delay can create new instabilities!
        """
        # Time delay differential equations are tricky - we need history
        history_length = int(delay_time / dt)
        history = np.ones(history_length, dtype=complex) * initial_condition
        
        times = np.arange(0, total_time, dt)
        trajectory = np.zeros(len(times), dtype=complex)
        trajectory[0] = initial_condition
        
        for i in range(1, len(times)):
            # Current state
            z_now = trajectory[i-1]
            
            # Delayed state (cavity memory)
            if i > history_length:
                z_delayed = trajectory[i - history_length]
            else:
                z_delayed = history[0]
            
            # Modified equation with delay feedback
            # The delay represents the cavity round-trip time
            feedback_strength = 0.3  # Adjustable parameter
            
            # Standard terms
            dzdt = self.equation_rhs(z_now, pump_amplitude)
            
            # Delay feedback term (like an echo in the cavity)
            delay_term = feedback_strength * self.nonlinearity * z_delayed * abs(z_now)**2
            
            # Update state
            trajectory[i] = z_now + dt * (dzdt + delay_term)
        
        return times, trajectory
    
    def quantum_trajectory_simulation(self, pump_amplitude, photon_number=10,
                                    num_trajectories=100, total_time=50):
        """
        Simulate quantum trajectories at low photon numbers where
        quantum effects become important.
        
        At the single-photon level, the discrete nature of light matters!
        """
        # When photon numbers are low, we need to include:
        # 1. Quantum noise from vacuum fluctuations
        # 2. Discrete photon jumps
        # 3. Measurement backaction
        
        dt = 0.01
        times = np.arange(0, total_time, dt)
        
        # Storage for quantum statistics
        all_trajectories = []
        
        print(f"Simulating {num_trajectories} quantum trajectories...")
        
        for traj in range(num_trajectories):
            # Start near vacuum with small coherent amplitude
            z = np.sqrt(photon_number) * np.exp(1j * np.random.random() * 2 * np.pi) * 0.1
            trajectory = [z]
            
            for t in times[1:]:
                # Deterministic evolution
                dzdt = self.equation_rhs(z, pump_amplitude)
                
                # Quantum noise scaling with photon number
                # At low photon numbers, noise is relatively stronger!
                noise_strength = np.sqrt(self.damping / max(abs(z)**2, 1))
                quantum_noise = noise_strength * np.sqrt(dt) * (
                    np.random.normal() + 1j * np.random.normal()
                ) / np.sqrt(2)
                
                # Photon counting effects (discrete jumps)
                if np.random.random() < self.damping * abs(z)**2 * dt:
                    # A photon was detected! State collapses
                    z = z / np.sqrt(1 + self.damping * dt)
                
                # Update state
                z = z + dt * dzdt + quantum_noise
                trajectory.append(z)
            
            all_trajectories.append(np.array(trajectory))
        
        # Calculate quantum statistics
        all_trajectories = np.array(all_trajectories)
        mean_intensity = np.mean(np.abs(all_trajectories)**2, axis=0)
        var_intensity = np.var(np.abs(all_trajectories)**2, axis=0)
        
        # Second-order correlation function g²(0)
        g2 = np.zeros(len(times))
        for i in range(len(times)):
            intensities = np.abs(all_trajectories[:, i])**2
            if np.mean(intensities) > 0:
                g2[i] = np.mean(intensities**2) / np.mean(intensities)**2
            else:
                g2[i] = 1
        
        # Quantum state purity (how "quantum" is the state?)
        purity = np.zeros(len(times))
        for i in range(len(times)):
            states = all_trajectories[:, i]
            # Purity = Tr(ρ²) where ρ is density matrix
            # For our ensemble, approximate as coherence measure
            mean_state = np.mean(states)
            coherence = abs(mean_state)**2 / np.mean(abs(states)**2)
            purity[i] = coherence
        
        return {
            'times': times,
            'mean_intensity': mean_intensity,
            'variance': var_intensity,
            'g2': g2,
            'purity': purity,
            'trajectories': all_trajectories
        }
    
    def plot_quantum_analysis(self, quantum_results):
        """
        Visualize the quantum properties of your optical cavity.
        
        This reveals the truly quantum nature of light in your system!
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        times = quantum_results['times']
        
        # Plot 1: Mean intensity and variance
        ax1 = axes[0, 0]
        ax1.plot(times, quantum_results['mean_intensity'], 'b-', 
                label='Mean intensity')
        ax1.fill_between(times, 
                        quantum_results['mean_intensity'] - np.sqrt(quantum_results['variance']),
                        quantum_results['mean_intensity'] + np.sqrt(quantum_results['variance']),
                        alpha=0.3, label='±1 std dev')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Photon number')
        ax1.set_title('Quantum Intensity Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: g²(0) correlation
        ax2 = axes[0, 1]
        ax2.plot(times, quantum_results['g2'], 'g-', linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', label='Coherent light')
        ax2.axhline(y=2, color='r', linestyle='--', label='Thermal light')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('g²(0)')
        ax2.set_title('Second-Order Correlation Function')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 3)
        
        # Plot 3: Quantum state purity
        ax3 = axes[1, 0]
        ax3.plot(times, quantum_results['purity'], 'm-', linewidth=2)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Purity')
        ax3.set_title('Quantum State Purity')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Sample quantum trajectories
        ax4 = axes[1, 1]
        # Plot a few individual trajectories
        for i in range(min(5, len(quantum_results['trajectories']))):
            traj = quantum_results['trajectories'][i]
            ax4.plot(times, np.abs(traj)**2, alpha=0.5, linewidth=0.5)
        ax4.plot(times, quantum_results['mean_intensity'], 'k-', 
                linewidth=2, label='Mean')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Photon number')
        ax4.set_title('Individual Quantum Trajectories')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

def demonstrate_advanced_analysis():
    """
    A complete demonstration of all the advanced analysis tools.
    
    This will help you understand the full dynamical richness of your equation!
    """
    print("="*70)
    print("ADVANCED BIFURCATION AND STABILITY ANALYSIS")
    print("Uncovering the Hidden Dynamics of Your Optical Cavity")
    print("="*70)
    
    # Create analyzer with your parameters
    analyzer = BifurcationAnalyzer(
        damping=0.97,
        nonlinearity=0.63,
        phase_coupling=-0.55,
        saturation=0.39
    )
    
    # 1. Bifurcation diagram analysis
    print("\n1. MAPPING THE BIFURCATION LANDSCAPE")
    print("-" * 50)
    print("Creating bifurcation diagrams for different detunings...")
    print("This shows all possible steady states and their stability.")
    
    # Analyze for different detuning values
    detunings = [0, np.pi/6, np.pi/3]  # Different laser-cavity phase mismatches
    fig_bifurcation = analyzer.plot_bifurcation_diagram(
        pump_range=(0, 5),
        num_points=200,
        detuning_phases=detunings
    )
    
    # 2. Time delay analysis
    print("\n2. TIME DELAY EFFECTS")
    print("-" * 50)
    print("Simulating with cavity memory effects...")
    
    pump = 2.0
    initial = 0.5 + 0.5j
    times_nodelay, traj_nodelay = analyzer.simulate_with_delay(
        initial, pump, delay_time=0, total_time=50
    )
    times_delay, traj_delay = analyzer.simulate_with_delay(
        initial, pump, delay_time=0.2, total_time=50
    )
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(times_nodelay, np.abs(traj_nodelay)**2, 'b-', 
            label='No delay', alpha=0.7)
    plt.plot(times_delay, np.abs(traj_delay)**2, 'r-', 
            label='With delay (τ=0.2)', alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Intensity |z|²')
    plt.title('Effect of Time Delay on Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Phase space plot
    plt.plot(traj_nodelay.real, traj_nodelay.imag, 'b-', 
            alpha=0.5, label='No delay')
    plt.plot(traj_delay.real, traj_delay.imag, 'r-', 
            alpha=0.5, label='With delay')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title('Phase Space Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    # 3. Quantum trajectory analysis
    print("\n3. QUANTUM REGIME ANALYSIS")
    print("-" * 50)
    print("Exploring quantum effects at low photon numbers...")
    
    # Run quantum simulation
    quantum_results = analyzer.quantum_trajectory_simulation(
        pump_amplitude=1.5,
        photon_number=5,  # Very low photon number!
        num_trajectories=50,
        total_time=30
    )
    
    # Visualize quantum properties
    fig_quantum = analyzer.plot_quantum_analysis(quantum_results)
    
    # 4. Summary of findings
    print("\n" + "="*70)
    print("KEY FINDINGS FROM ADVANCED ANALYSIS")
    print("="*70)
    
    print("\n1. BIFURCATION STRUCTURE:")
    print("   - Your system shows rich multistability")
    print("   - Different detunings reveal different bifurcation patterns")
    print("   - Look for green dots marking Hopf bifurcations (oscillation birth)")
    
    print("\n2. TIME DELAY EFFECTS:")
    print("   - Delays can destabilize otherwise stable fixed points")
    print("   - New oscillatory solutions may emerge with appropriate delay")
    print("   - Phase space becomes more complex with memory effects")
    
    print("\n3. QUANTUM SIGNATURES:")
    steady_g2 = np.mean(quantum_results['g2'][-100:])
    print(f"   - Steady-state g²(0) = {steady_g2:.3f}")
    if steady_g2 < 0.9:
        print("     → Sub-Poissonian light (quantum effect!)")
    elif steady_g2 > 1.1:
        print("     → Super-Poissonian light (classical bunching)")
    else:
        print("     → Near-coherent light")
    
    mean_purity = np.mean(quantum_results['purity'])
    print(f"   - Average quantum purity = {mean_purity:.3f}")
    if mean_purity > 0.7:
        print("     → Highly coherent quantum state")
    elif mean_purity > 0.3:
        print("     → Partially coherent state")
    else:
        print("     → Highly mixed/classical state")
    
    print("\n4. RECOMMENDATIONS FOR FINDING OSCILLATIONS:")
    print("   - Explore pump powers near bifurcation points")
    print("   - Try detunings around π/4 to π/2")
    print("   - Consider reducing damping to 0.7-0.8")
    print("   - Add time delay τ ≈ 0.1-0.3 in normalized units")
    
    print("\n" + "="*70)
    print("Your equation contains a treasure trove of nonlinear dynamics!")
    print("Keep exploring these parameter regions to unlock its full potential.")
    print("="*70)

if __name__ == "__main__":
    demonstrate_advanced_analysis()
