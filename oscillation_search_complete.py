
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class OscillationEnhancedAnalysis:
    """
    Modified analysis to promote and detect oscillatory behavior in the quantum-classical system.
    """
    
    def __init__(self):
        # Original equation parameters - these are suppressing oscillations
        self.original_coeffs = {
            'linear': -0.97,      # Strong damping kills oscillations!
            'cubic': 0.63,
            'phase': -0.55,
            'normalization': -0.39
        }
        
        # Modified parameters to enhance oscillations
        self.oscillation_coeffs = {
            'linear': -0.1,       # Much weaker damping
            'cubic': 0.3,         # Moderate nonlinearity
            'phase': -0.8,        # Stronger phase coupling
            'normalization': -0.2  # Weaker normalization
        }
    
    def equation_func(self, z, c, use_oscillation_params=True):
        """
        The dynamical equation with switchable parameters.
        """
        if use_oscillation_params:
            coeffs = self.oscillation_coeffs
        else:
            coeffs = self.original_coeffs
        
        # Avoid division by zero
        z_norm = z / np.maximum(abs(z), 0.01)
        
        result = (coeffs['linear'] * z + 
                 coeffs['cubic'] * z**3 + 
                 coeffs['phase'] * np.exp(1j * c.real) * z + 
                 coeffs['normalization'] * z_norm)
        
        return result
    
    def detect_oscillations_improved(self, z0, c, max_iter=1000, use_oscillation_params=True):
        """
        Improved oscillation detection with multiple methods.
        """
        trajectory = []
        z = z0
        
        # Generate trajectory
        for i in range(max_iter):
            z = self.equation_func(z, c, use_oscillation_params)
            trajectory.append(z)
            
            if abs(z) > 50:  # Escaped
                return {
                    'type': 'escape',
                    'oscillatory': False,
                    'period': None,
                    'amplitude': None
                }
        
        # Convert to arrays for analysis
        trajectory = np.array(trajectory)
        magnitudes = np.abs(trajectory)
        phases = np.angle(trajectory)
        
        # Method 1: Check for periodic returns in magnitude
        if len(trajectory) > 100:
            # Look at the last half to avoid transients
            mag_signal = magnitudes[len(magnitudes)//2:]
            
            # Find peaks in magnitude
            peaks, properties = find_peaks(mag_signal, distance=2)
            
            if len(peaks) >= 3:
                # Check if peaks are regularly spaced
                peak_distances = np.diff(peaks)
                if np.std(peak_distances) < 0.3 * np.mean(peak_distances):
                    # Regular oscillations detected!
                    period = np.mean(peak_distances)
                    amplitude = np.std(mag_signal)
                    
                    return {
                        'type': 'oscillations',
                        'oscillatory': True,
                        'period': period,
                        'amplitude': amplitude,
                        'frequency': 1.0 / period if period > 0 else 0
                    }
        
        # Method 2: Check for rotation in complex plane
        if len(trajectory) > 50:
            # Calculate winding number
            phase_diff = np.diff(phases)
            # Handle phase wrapping
            phase_diff = np.where(phase_diff > np.pi, phase_diff - 2*np.pi, phase_diff)
            phase_diff = np.where(phase_diff < -np.pi, phase_diff + 2*np.pi, phase_diff)
            
            total_rotation = np.sum(phase_diff)
            rotations = abs(total_rotation) / (2 * np.pi)
            
            if rotations > 2:  # At least 2 full rotations
                return {
                    'type': 'spiral_oscillations',
                    'oscillatory': True,
                    'period': len(trajectory) / rotations,
                    'amplitude': np.std(magnitudes),
                    'rotations': rotations
                }
        
        # Method 3: Check for limit cycles
        if len(trajectory) > 200:
            # Look for returns to neighborhood
            ref_point = trajectory[100]  # Reference after transients
            distances = np.abs(trajectory[101:] - ref_point)
            
            close_returns = np.where(distances < 0.1)[0]
            if len(close_returns) > 2:
                return_times = np.diff(close_returns)
                if len(return_times) > 0 and np.std(return_times) < 0.5 * np.mean(return_times):
                    return {
                        'type': 'limit_cycle',
                        'oscillatory': True,
                        'period': np.mean(return_times),
                        'amplitude': np.std(magnitudes[100:])
                    }
        
        # Check if it's a fixed point
        if np.std(magnitudes[-50:]) < 0.01:
            return {
                'type': 'fixed_point',
                'oscillatory': False,
                'period': None,
                'amplitude': 0
            }
        
        # Otherwise, it's complex non-oscillatory behavior
        return {
            'type': 'complex',
            'oscillatory': False,
            'period': None,
            'amplitude': np.std(magnitudes)
        }
    
    def scan_parameter_space_for_oscillations(self, damping_range=(-0.5, 0), 
                                            phase_range=(-2, -0.1), 
                                            resolution=50):
        """
        Scan parameter space specifically looking for oscillations.
        """
        damping_vals = np.linspace(damping_range[0], damping_range[1], resolution)
        phase_vals = np.linspace(phase_range[0], phase_range[1], resolution)
        
        oscillation_map = np.zeros((resolution, resolution))
        period_map = np.zeros((resolution, resolution))
        
        # We'll modify the coefficients during the scan
        original_linear = self.oscillation_coeffs['linear']
        original_phase = self.oscillation_coeffs['phase']
        
        for i, phase_coeff in enumerate(phase_vals):
            for j, damping in enumerate(damping_vals):
                # Update coefficients
                self.oscillation_coeffs['linear'] = damping
                self.oscillation_coeffs['phase'] = phase_coeff
                
                # Test with multiple initial conditions
                oscillation_found = False
                min_period = np.inf
                
                for z0 in [0.1+0.1j, 0.5+0j, 0+0.5j, 0.3+0.3j]:
                    c = complex(0.5, 0.5)  # Fixed c for this scan
                    
                    result = self.detect_oscillations_improved(z0, c, max_iter=500)
                    
                    if result['oscillatory']:
                        oscillation_found = True
                        if result['period'] is not None and result['period'] < min_period:
                            min_period = result['period']
                
                oscillation_map[i, j] = 1 if oscillation_found else 0
                period_map[i, j] = min_period if oscillation_found else 0
        
        # Restore original coefficients
        self.oscillation_coeffs['linear'] = original_linear
        self.oscillation_coeffs['phase'] = original_phase
        
        return oscillation_map, period_map, damping_vals, phase_vals
    
    def visualize_oscillation_regions(self, save_path='oscillation_regions.png'):
        """
        Create visualization of parameter regions that support oscillations.
        """
        print("Scanning parameter space for oscillatory regions...")
        print("This explores damping and phase coupling effects on oscillations")
        
        # Scan with parameters more likely to show oscillations
        osc_map, period_map, damp_vals, phase_vals = self.scan_parameter_space_for_oscillations(
            damping_range=(-0.3, 0),      # Weak damping to no damping
            phase_range=(-1.5, -0.3),     # Moderate to strong phase coupling
            resolution=40
        )
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot 1: Oscillation regions
        im1 = ax1.imshow(osc_map, extent=[damp_vals[0], damp_vals[-1], 
                                          phase_vals[0], phase_vals[-1]],
                        origin='lower', aspect='auto', cmap='RdBu')
        ax1.set_xlabel('Damping Coefficient')
        ax1.set_ylabel('Phase Coupling Coefficient')
        ax1.set_title('Oscillatory Regions (Blue = Oscillations)')
        ax1.grid(True, alpha=0.3)
        
        # Add contour lines
        ax1.contour(damp_vals, phase_vals, osc_map, levels=[0.5], colors='black', linewidths=2)
        
        # Plot 2: Period map
        period_masked = np.ma.masked_where(period_map == 0, period_map)
        im2 = ax2.imshow(period_masked, extent=[damp_vals[0], damp_vals[-1], 
                                               phase_vals[0], phase_vals[-1]],
                        origin='lower', aspect='auto', cmap='viridis')
        ax2.set_xlabel('Damping Coefficient')
        ax2.set_ylabel('Phase Coupling Coefficient')
        ax2.set_title('Oscillation Period')
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(im2, ax=ax2, label='Period')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        osc_percentage = np.sum(osc_map) / osc_map.size * 100
        print(f"\nOscillatory regions: {osc_percentage:.1f}% of parameter space")
        print(f"Average period in oscillatory regions: {np.mean(period_map[period_map > 0]):.1f}")
        
        return osc_map, period_map
    
    def demonstrate_oscillations(self):
        """
        Show specific examples of oscillatory behavior.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Test cases with different parameters
        test_cases = [
            {'damping': -0.05, 'phase': -0.8, 'title': 'Weak damping, strong phase'},
            {'damping': -0.1, 'phase': -1.2, 'title': 'Moderate damping, very strong phase'},
            {'damping': 0.0, 'phase': -0.6, 'title': 'No damping, moderate phase'},
            {'damping': -0.15, 'phase': -0.9, 'title': 'Original-like but weaker'},
            {'damping': -0.02, 'phase': -1.0, 'title': 'Minimal damping'},
            {'damping': -0.08, 'phase': -0.7, 'title': 'Balanced parameters'}
        ]
        
        for idx, (ax, test) in enumerate(zip(axes.flat, test_cases)):
            # Set parameters
            self.oscillation_coeffs['linear'] = test['damping']
            self.oscillation_coeffs['phase'] = test['phase']
            
            # Generate trajectory
            z0 = complex(0.5, 0.5)
            c = complex(0.5, 0)
            trajectory = []
            z = z0
            
            for _ in range(300):
                z = self.equation_func(z, c)
                trajectory.append(z)
            
            trajectory = np.array(trajectory)
            
            # Plot in complex plane
            ax.plot(trajectory.real, trajectory.imag, 'b-', alpha=0.6, linewidth=0.8)
            ax.plot(trajectory[0].real, trajectory[0].imag, 'go', markersize=8, label='Start')
            ax.plot(trajectory[-1].real, trajectory[-1].imag, 'ro', markersize=8, label='End')
            
            # Detect oscillation type
            result = self.detect_oscillations_improved(z0, c)
            
            ax.set_title(f"{test['title']}\n{result['type']}", fontsize=10)
            ax.set_xlabel('Real')
            ax.set_ylabel('Imaginary')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            if idx == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig('oscillation_examples.png', dpi=150, bbox_inches='tight')
        plt.show()

def suggest_parameter_modifications():
    """
    Provide specific suggestions for getting oscillations in your system.
    """
    print("\n" + "="*60)
    print("PARAMETER MODIFICATION SUGGESTIONS FOR OSCILLATIONS")
    print("="*60)
    
    print("\n1. REDUCE DAMPING:")
    print("   Your current damping coefficient is -0.97 (very strong!)")
    print("   Try reducing to -0.1 to -0.3 range")
    print("   → Weak damping allows sustained oscillations")
    
    print("\n2. ADJUST PHASE COUPLING:")
    print("   Current: -0.55")
    print("   Try: -0.8 to -1.2")
    print("   → Stronger phase coupling drives rotation in complex plane")
    
    print("\n3. MODIFY NONLINEARITY:")
    print("   Current cubic coefficient: 0.63")
    print("   Try: 0.2 to 0.4")
    print("   → Moderate nonlinearity prevents runaway growth")
    
    print("\n4. INITIAL CONDITIONS:")
    print("   Try z0 away from origin (e.g., 0.5+0.5j)")
    print("   → Avoids singular behavior of z/|z| term")
    
    print("\n5. PARAMETER COMBINATIONS FOR OSCILLATIONS:")
    print("   • (damping=-0.1, phase=-1.0): Limit cycles")
    print("   • (damping=-0.05, phase=-0.8): Quasi-periodic")
    print("   • (damping=0, phase=-0.6): Pure oscillations")

# Run the enhanced analysis
if __name__ == "__main__":
    analyzer = OscillationEnhancedAnalysis()
    
    print("ENHANCED OSCILLATION ANALYSIS")
    print("-" * 40)
    
    # Show parameter suggestions
    suggest_parameter_modifications()
    
    # Visualize oscillation regions
    print("\n\nGenerating oscillation region maps...")
    analyzer.visualize_oscillation_regions()
    
    # Show specific examples
    print("\nGenerating oscillation examples...")
    analyzer.demonstrate_oscillations()
    
    print("\n✓ Analysis complete! Check the generated images.")
    print("  - oscillation_regions.png: Parameter space map")
    print("  - oscillation_examples.png: Specific trajectory examples")
