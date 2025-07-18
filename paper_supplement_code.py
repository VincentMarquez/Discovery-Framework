#!/usr/bin/env python3
"""
Supplementary Code: Parameter Sensitivity Analysis
Demonstrates how recoil direction fundamentally changes dynamics

Paper: A Physics-Inspired Design Paradigm for Novel Dynamics in Non-Holomorphic Maps
Section: Parameter Sensitivity Analysis
"""

import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt

def f(z, a, b, k):
    """The magnitude-recoil map"""
    if abs(z) < 1e-15:
        return 0.0
    return a * z + b * z**3 - k * z / abs(z)

def analyze_parameter_set(a, b, k, label=""):
    """Complete dynamical analysis for one parameter set"""
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {label}")
    print(f"Parameters: a={a}, b={b}, k={k}")
    print(f"{'='*70}")
    
    results = {
        'parameters': {'a': a, 'b': b, 'k': k},
        'label': label,
        'timestamp': datetime.now().isoformat()
    }
    
    # 1. Fixed Point / Periodic Orbit Analysis
    print("\n1. FIXED POINT/PERIODIC ORBIT ANALYSIS")
    print("-" * 40)
    
    # Test for period-1 fixed points
    fixed_points = []
    for r_test in np.linspace(0.1, 2.0, 20):
        for sign in [1, -1]:
            z = sign * r_test
            z_new = f(z, a, b, k)
            if abs(z_new - z) < 1e-6:
                fixed_points.append(z)
                print(f"Fixed point found: z = {z:.3f}")
    
    # Test for period-2 cycle
    period2_points = []
    for r_test in np.linspace(0.1, 2.0, 20):
        for sign in [1, -1]:
            z0 = sign * r_test
            z1 = f(z0, a, b, k)
            z2 = f(z1, a, b, k)
            if abs(z2 - z0) < 1e-6 and abs(z1 - z0) > 1e-6:
                if not any(abs(z0 - p) < 1e-3 for p, _ in period2_points):
                    period2_points.append((z0, z1))
                    print(f"Period-2 cycle: z₁ = {z0:.3f} ↔ z₂ = {z1:.3f}")
                    
                    # Check stability
                    eps = 1e-6
                    df0 = abs(f(z0 + eps, a, b, k) - f(z0 - eps, a, b, k)) / (2 * eps)
                    df1 = abs(f(z1 + eps, a, b, k) - f(z1 - eps, a, b, k)) / (2 * eps)
                    stability = df0 * df1
                    print(f"  Stability (|f'(z₁)·f'(z₂)|): {stability:.3f}")
                    print(f"  → {'Stable' if stability < 1 else 'Unstable'}")
    
    results['fixed_points'] = len(fixed_points)
    results['period2_cycles'] = len(period2_points)
    
    # 2. Lyapunov Exponent Calculation
    print("\n2. LYAPUNOV EXPONENT ANALYSIS")
    print("-" * 40)
    
    lyapunov_values = []
    n_orbits = 100
    
    for _ in range(n_orbits):
        r0 = 0.5 + 1.5 * np.random.rand()
        theta0 = 2 * np.pi * np.random.rand()
        z = r0 * np.exp(1j * theta0)
        
        z_shadow = z * (1 + 1e-8)
        lyap_sum = 0
        valid_steps = 0
        
        for t in range(1000):
            z_new = f(z, a, b, k)
            z_shadow_new = f(z_shadow, a, b, k)
            
            if abs(z_new) > 50:
                break
                
            d0 = abs(z_shadow - z)
            d1 = abs(z_shadow_new - z_new)
            
            if d0 > 0 and d1 > 0:
                lyap_sum += np.log(d1/d0)
                valid_steps += 1
            
            if d1 > 1e-5:
                z_shadow_new = z_new * (1 + 1e-8)
            
            z = z_new
            z_shadow = z_shadow_new
        
        if valid_steps > 100:
            lyapunov_values.append(lyap_sum / valid_steps)
    
    if lyapunov_values:
        lyap_mean = np.mean(lyapunov_values)
        lyap_std = np.std(lyapunov_values)
        results['lyapunov'] = {'mean': lyap_mean, 'std': lyap_std}
        print(f"Mean Lyapunov exponent: {lyap_mean:.3f} ± {lyap_std:.3f}")
        print(f"% positive: {100*sum(l > 0 for l in lyapunov_values)/len(lyapunov_values):.1f}%")
    
    # 3. Basin Structure Analysis
    print("\n3. BASIN STRUCTURE ANALYSIS")
    print("-" * 40)
    
    n_test = 1000
    outcomes = {'escaped': 0, 'converged': 0, 'bounded': 0}
    escape_times = []
    
    for _ in range(n_test):
        r0 = 0.5 + 2.0 * np.random.rand()
        z = r0 * np.exp(2j * np.pi * np.random.rand())
        
        for t in range(10000):
            z_old = z
            z = f(z, a, b, k)
            
            if abs(z) > 50:
                outcomes['escaped'] += 1
                escape_times.append(t)
                break
            elif t > 100 and abs(z - z_old) < 1e-8:
                outcomes['converged'] += 1
                escape_times.append(10000)
                break
        else:
            outcomes['bounded'] += 1
            escape_times.append(10000)
    
    results['basin_structure'] = {
        'escape_rate': outcomes['escaped'] / n_test,
        'convergence_rate': outcomes['converged'] / n_test,
        'bounded_rate': outcomes['bounded'] / n_test,
        'mean_transient': np.mean(escape_times)
    }
    
    print(f"Escape to infinity: {100*outcomes['escaped']/n_test:.1f}%")
    print(f"Converge to attractor: {100*outcomes['converged']/n_test:.1f}%")
    print(f"Still bounded at t=10000: {100*outcomes['bounded']/n_test:.1f}%")
    print(f"Mean transient length: {np.mean(escape_times):.0f}")
    
    # 4. Sample Orbit Evolution
    print("\n4. SAMPLE ORBIT EVOLUTION")
    print("-" * 40)
    
    z = 1.0 + 0.0j
    print(f"Starting at z = {z}")
    trajectory = []
    
    for t in range(20):
        z = f(z, a, b, k)
        trajectory.append(abs(z))
        if t < 10 or t % 5 == 0:
            print(f"  t={t+1}: |z| = {abs(z):.6f}")
    
    results['sample_trajectory'] = trajectory
    
    return results

def create_comparison_plot(results_list):
    """Create visual comparison of dynamics"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    for idx, results in enumerate(results_list):
        ax_row = idx
        
        # Plot 1: Sample trajectory
        ax = axes[ax_row, 0]
        traj = results['sample_trajectory']
        ax.plot(traj, 'b-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('|z|')
        ax.set_title(f"{results['label']}: |z| evolution")
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Basin statistics
        ax = axes[ax_row, 1]
        labels = ['Escape', 'Converge', 'Bounded']
        sizes = [
            results['basin_structure']['escape_rate'],
            results['basin_structure']['convergence_rate'],
            results['basin_structure']['bounded_rate']
        ]
        colors = ['red', 'green', 'blue']
        
        # Only plot non-zero values
        plot_data = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
        if plot_data:
            labels, sizes, colors = zip(*plot_data)
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            ax.set_title(f"{results['label']}: Basin structure")
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_comparison.png', dpi=150)
    print("\n✓ Saved comparison plot to parameter_sensitivity_comparison.png")

def main():
    """Run complete parameter sensitivity analysis"""
    
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("Magnitude-Recoil Map: f(z) = az + bz³ - k(z/|z|)")
    
    # Analyze both parameter sets
    results_original = analyze_parameter_set(
        a=-0.97, b=0.63, k=-0.39,
        label="Original (k=-0.39, outward recoil)"
    )
    
    results_inverted = analyze_parameter_set(
        a=-0.97, b=0.63, k=+0.39,
        label="Inverted (k=+0.39, inward recoil)"
    )
    
    # Create comparison plot
    create_comparison_plot([results_original, results_inverted])
    
    # Save all results
    all_results = {
        'analysis_type': 'parameter_sensitivity',
        'original': results_original,
        'inverted': results_inverted
    }
    
    with open('parameter_sensitivity_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✓ Complete results saved to parameter_sensitivity_results.json")
    
    # Print summary comparison
    print("\n" + "="*70)
    print("SUMMARY COMPARISON")
    print("="*70)
    print(f"{'Property':<30} {'k=-0.39':<20} {'k=+0.39':<20}")
    print("-"*70)
    
    # Attractors
    print(f"{'Fixed points':<30} "
          f"{results_original['fixed_points']:<20} "
          f"{results_inverted['fixed_points']:<20}")
    
    print(f"{'Period-2 cycles':<30} "
          f"{results_original['period2_cycles']:<20} "
          f"{results_inverted['period2_cycles']:<20}")
    
    # Lyapunov
    if 'lyapunov' in results_original:
        lyap_orig = f"{results_original['lyapunov']['mean']:.3f}"
    else:
        lyap_orig = "N/A"
        
    if 'lyapunov' in results_inverted:
        lyap_inv = f"{results_inverted['lyapunov']['mean']:.3f}"
    else:
        lyap_inv = "N/A"
        
    print(f"{'Mean Lyapunov exponent':<30} {lyap_orig:<20} {lyap_inv:<20}")
    
    # Basin structure
    esc_orig = f"{100*results_original['basin_structure']['escape_rate']:.1f}%"
    esc_inv = f"{100*results_inverted['basin_structure']['escape_rate']:.1f}%"
    
    print(f"{'Escape rate':<30} {esc_orig:<20} {esc_inv:<20}")
    
    print("\n✓ Analysis complete!")

if __name__ == "__main__":
    main()
