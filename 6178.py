import numpy as np
import logging
from datetime import datetime
import random
from statistics import mean, stdev
import os
import pickle

# Set up logging
log_filename = f"emergent_novel_math_seeded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Checkpoint file to save progress
checkpoint_file = "checkpoint.pkl"

def save_checkpoint(experiment_results, current_exp):
    """Save progress to a checkpoint file."""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'experiment_results': experiment_results, 'current_exp': current_exp}, f)

def load_checkpoint():
    """Load progress from a checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        return data['experiment_results'], data['current_exp']
    return [], 0

# Define the 10 novel equations as seeds with 'experiment' key
seed_equations = [
    # Experiment 6178
    {
        "experiment": 6178,
        "terms": [
            ("z", lambda z, c: z, "Classical: Linear term"),
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("exp(1j*c.real)*z", lambda z, c: np.exp(1j * c.real) * z, "Quantum: Phase from real part"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state")
        ],
        "coeffs": [-0.97, 0.63, -0.55, -0.39]
    },
    # Experiment 1253
    {
        "experiment": 1253,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("c", lambda z, c: c, "Classical: External parameter"),
            ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude"),
            ("exp(1j*c.real)*z", lambda z, c: np.exp(1j * c.real) * z, "Quantum: Phase from real part")
        ],
        "coeffs": [0.74, -0.70, 0.95, 0.68]
    },
    # Experiment 1400
    {
        "experiment": 1400,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("c", lambda z, c: c, "Classical: External parameter"),
            ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude")
        ],
        "coeffs": [0.84, -0.96, 0.89]
    },
    # Experiment 4412
    {
        "experiment": 4412,
        "terms": [
            ("z**2", lambda z, c: z**2, "Classical: Quadratic nonlinearity"),
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("exp(1j*c.real)*z", lambda z, c: np.exp(1j * c.real) * z, "Quantum: Phase from real part"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state")
        ],
        "coeffs": [-0.79, -0.77, 0.97, 0.16]
    },
    # Experiment 9105
    {
        "experiment": 9105,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("z**2", lambda z, c: z**2, "Classical: Quadratic nonlinearity"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state")
        ],
        "coeffs": [0.80, 0.89, -0.99]
    },
    # Experiment 4772
    {
        "experiment": 4772,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state"),
            ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude")
        ],
        "coeffs": [0.77, 0.79, 0.82]
    },
    # Experiment 9497
    {
        "experiment": 9497,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state"),
            ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude")
        ],
        "coeffs": [-0.70, 0.30, 0.90]
    },
    # Experiment 485
    {
        "experiment": 485,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("z**2", lambda z, c: z**2, "Classical: Quadratic nonlinearity"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state"),
            ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude")
        ],
        "coeffs": [-0.85, -0.43, 0.90, -0.28]
    },
    # Experiment 9630
    {
        "experiment": 9630,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("z", lambda z, c: z, "Classical: Linear term"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state"),
            ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude")
        ],
        "coeffs": [0.61, -0.44, -0.84, -0.15]
    },
    # Experiment 15
    {
        "experiment": 15,
        "terms": [
            ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
            ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude"),
            ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state")
        ],
        "coeffs": [-0.89, 0.80, -0.90]
    }
]

def generate_variation(seed):
    """Generate a variation of a seed equation by perturbing coefficients and terms."""
    classical_terms = [
        ("z**2", lambda z, c: z**2, "Classical: Quadratic nonlinearity"),
        ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
        ("z", lambda z, c: z, "Classical: Linear term"),
        ("c", lambda z, c: c, "Classical: External parameter")
    ]
    quantum_terms = [
        ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude"),
        ("exp(1j*c.real)*z", lambda z, c: np.exp(1j * c.real) * z, "Quantum: Phase from real part"),
        ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state")
    ]
    all_possible_terms = classical_terms + quantum_terms

    # Start with the seed terms and coefficients
    terms = seed["terms"].copy()
    coeffs = seed["coeffs"].copy()

    # Perturb coefficients
    coeffs = [coeff + random.uniform(-0.1, 0.1) for coeff in coeffs]

    # Randomly add or remove a term (50% chance to add, 50% to remove if possible)
    if random.random() < 0.5 and len(terms) > 1:  # Remove a term
        remove_idx = random.randint(0, len(terms) - 1)
        terms.pop(remove_idx)
        coeffs.pop(remove_idx)
    else:  # Add a term
        available_terms = [t for t in all_possible_terms if t not in terms]
        if available_terms:
            new_term = random.choice(available_terms)
            terms.append(new_term)
            coeffs.append(random.uniform(-1, 1))

    # Define the equation function
    def equation(z, c):
        term_contribs = {}
        result = 0
        for (term, term_func, _), coeff in zip(terms, coeffs):
            contrib = coeff * term_func(z, c)
            term_contribs[term] = contrib
            result += contrib
        if np.isnan(result) or np.isinf(result):
            return z, term_contribs
        return result, term_contribs

    equation_str = " + ".join(f"{coeff:.2f}*{term}" for (term, _, _), coeff in zip(terms, coeffs))
    equation_desc = "; ".join(f"{coeff:.2f}*{desc}" for (_, _, desc), coeff in zip(terms, coeffs))
    return equation, equation_str, equation_desc, terms, coeffs

def generate_fractal(width, height, x_min, x_max, y_min, y_max, max_iter, equation_func, equation_name, all_terms, coeffs, experiment_num):
    """Compute iterations and track numbers for sample points with initial z and term contributions."""
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    fractal = np.zeros((height, width))
    c_metrics = {}  # Track avg |z| per c
    
    # Exact sample points within range
    sample_cs = [
        complex(-0.5, 0.0),  # Center
        complex(-2.0, 1.5),  # Top-left
        complex(1.0, -1.5)   # Bottom-right
    ]
    sample_histories = {f"c={c:.2f}": [] for c in sample_cs}
    term_histories = {f"c={c:.2f}": [] for c in sample_cs}  # Store term contributions

    # Print and log for first 5 experiments only
    if experiment_num <= 5:
        print(f"\nExperiment {experiment_num} for: z_(n+1) = {equation_name}")
        print(f"Description: {equation_desc}")
        logging.info(f"\nExperiment {experiment_num}: z_(n+1) = {equation_name}")
        logging.info(f"Description: {equation_desc}")

    for i in range(height):
        for j in range(width):
            c = x[j] + y[i]*1j
            z = complex(random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01))
            z_history = [abs(z)]  # Track |z| for c metrics
            for k in range(max_iter):
                z_prev = z
                z, term_contribs = equation_func(z, c)
                z_history.append(abs(z))
                if c in [s.real + s.imag*1j for s in sample_cs]:
                    c_key = f"c={c:.2f}"
                    sample_histories[c_key].append(z)
                    term_histories[c_key].append(term_contribs)
                    if experiment_num <= 5 and len(sample_histories[c_key]) <= 5:
                        print(f"{c_key}, Iter {k}: z = {z.real:.3f} + {z.imag:.3f}i, |z| = {abs(z):.3f}")
                        logging.info(f"{c_key}, Iter {k}: z = {z.real:.3f} + {z.imag:.3f}i, |z| = {abs(z):.3f}")
                        for term, contrib in term_contribs.items():
                            logging.info(f"  Term {term} contribution: {contrib.real:.3f} + {contrib.imag:.3f}i")
                if abs(z) > 2 or z == z_prev:  # Detect stall
                    fractal[i, j] = k
                    break
            else:
                fractal[i, j] = max_iter
            c_metrics[c] = np.mean(z_history)  # Avg |z| for this c
    
    return fractal, sample_histories, term_histories, c_metrics

def log_numbers(history, term_history, equation_name, sample_c, experiment_num):
    """Log detailed iteration numbers and term contributions for first 5 experiments."""
    if experiment_num <= 5:
        logging.info(f"Sample point {sample_c} under {equation_name}:")
        for i, (z, terms) in enumerate(zip(history, term_history)):
            logging.info(f"Iter {i}: z = {z.real:.3f} + {z.imag:.3f}i, |z| = {abs(z):.3f}")
            for term, contrib in terms.items():
                logging.info(f"  Term {term} contribution: {contrib.real:.3f} + {contrib.imag:.3f}i")
        if len(history) < max_iter:
            logging.info(f"Diverged or stalled after {len(history)} iterations.")
        else:
            logging.info(f"Bounded up to {max_iter} iterations.")

def detect_emergence(history):
    """Detect emergent patterns in z evolution."""
    if len(history) < 5:  # Increase to 5 iterations
        return "No emergence detectable."
    magnitudes = [abs(z) for z in history]
    diffs = np.diff(magnitudes)
    # Check for near-cycles
    if len(diffs) > 3 and np.any(np.abs(diffs[2:] - diffs[:-2]) < 0.002):  # Relaxed oscillation
        return "Emergent: Possible oscillation or near-cycle detected."
    # Check for slow convergence
    if len(diffs) > 5 and np.any(np.abs(diffs[-5:]) < 0.0005):  # Relaxed convergence
        return "Emergent: Slow convergence to fixed point."
    # Check for fixed points
    if len(diffs) > 3 and all(abs(d) < 0.0001 for d in diffs[-3:]):  # Fixed point
        return "Emergent: Fixed point detected."
    # Check for amplitude shifts
    if max(magnitudes) - min(magnitudes) > 0.2 and len(history) > 5:  # Lowered threshold
        return "Emergent: Significant amplitude shift observed."
    # Check for unexpected stability
    if all(m < 1.0 for m in magnitudes) and len(history) == max_iter:
        return "Emergent: Unexpected stability despite classical terms."
    return "No clear emergent pattern."

def estimate_fractal_dimension(fractal):
    """Basic box-counting dimension estimate (simplified)."""
    if fractal.size == 0:
        return 0
    threshold = np.mean(fractal)  # Use mean iteration as threshold
    binary_fractal = (fractal > threshold).astype(int)
    box_sizes = [2, 4, 8, 16]  # Box sizes to count
    counts = []
    for size in box_sizes:
        boxes = 0
        for i in range(0, binary_fractal.shape[0], size):
            for j in range(0, binary_fractal.shape[1], size):
                if np.any(binary_fractal[i:i+size, j:j+size]):
                    boxes += 1
        counts.append(boxes)
    if len(counts) < 2 or all(c == 0 for c in counts):
        return 0
    log_counts = np.log(counts)
    log_sizes = np.log(box_sizes)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)  # Dimension = -slope
    return -slope if -slope > 0 else 0

def compute_novelty_score(history, avg_iter, mean_iter, max_iter, fractal_dim):
    """Score novelty based on deviation, emergent behavior, and fractal dimension."""
    if len(history) < 3:
        return 0
    magnitudes = [abs(z) for z in history]
    oscillation = stdev(magnitudes) if len(magnitudes) > 1 else 0
    stability = avg_iter / max_iter  # Normalize to 0-1
    deviation = abs(avg_iter - mean_iter) / max_iter  # Normalize deviation
    emergence_bonus = 0.6 if "Emergent" in detect_emergence(history) else 0
    term_diversity = len(magnitudes) / max_iter  # Normalize by max iterations
    fractal_bonus = min(fractal_dim / 2.0, 0.2)  # Cap fractal contribution
    return 0.25 * oscillation + 0.2 * stability + 0.2 * deviation + 0.15 * term_diversity + emergence_bonus + fractal_bonus

# Main execution
if __name__ == "__main__":
    # Optimized settings for Pydroid 3 with adjustment based on visualization
    width = 50  # Increase to 50x50 for better resolution around the transition zone
    height = 50
    x_min, x_max = -0.5, 0.5  # Zoom in on the transition zone from the visualization
    y_min, y_max = -0.5, 0.5
    max_iter = 100  # Increased to 100 for deeper dynamics
    num_experiments = 10000  # Set to 10,000 as requested
    experiments_per_seed = num_experiments // len(seed_equations)  # 1,000 per seed

    print("Starting classical-quantum fusion for 10,000 experiments (seeded from 10 novel equations).")
    print(f"Grid: {width}x{height}, Max iters: {max_iter}, Range: Re [{x_min}, {x_max}], Im [{y_min}, {y_max}]")
    logging.info("Starting classical-quantum fusion for 10,000 experiments (seeded from 10 novel equations).")
    logging.info(f"Grid: {width}x{height}, Max iters: {max_iter}, Range: Re [{x_min}, {x_max}], Im [{y_min}, {y_max}]")

    # Load checkpoint if exists
    experiment_results, start_exp = load_checkpoint()
    if start_exp > 0:
        print(f"Resuming from experiment {start_exp}...")
    else:
        experiment_results = []

    center_c = complex(-0.5, 0.0)  # Consistent center point
    center_histories = {}  # Store only center point history
    c_metrics = {}  # Track c-specific metrics

    for i in range(start_exp, num_experiments):
        # Select the seed equation (cycle through the 10 seeds)
        seed_idx = (i // experiments_per_seed) % len(seed_equations)
        seed = seed_equations[seed_idx]

        # Generate a variation of the seed equation
        try:
            equation_func, equation_name, equation_desc, all_terms, coeffs = generate_variation(seed)
        except Exception as e:
            logging.error(f"Error generating equation for experiment {i+1}: {str(e)}")
            continue

        # Generate fractal and track numbers
        try:
            fractal, sample_histories, term_histories, c_metrics = generate_fractal(
                width, height, x_min, x_max, y_min, y_max, max_iter, equation_func, 
                equation_name, all_terms, coeffs, i+1
            )
        except (MemoryError, Exception) as e:
            logging.error(f"Error in experiment {i+1}: {str(e)}")
            continue

        # Log detailed numbers and terms only for first 5 experiments
        if i <= 5:
            for sample_c, history in sample_histories.items():
                log_numbers(history, term_histories[sample_c], equation_name, sample_c, i+1)

        # Compute metrics
        avg_iter = np.mean(fractal)
        center_history = sample_histories.get(f"c={center_c:.2f}", [])
        center_histories[i] = center_history  # Store only center history
        fractal_dim = estimate_fractal_dimension(fractal)
        emergence = detect_emergence(center_history)
        behavior_score = compute_novelty_score(center_history, avg_iter, 0, max_iter, fractal_dim)

        # Store results
        experiment_results.append({
            "experiment": i+1,
            "equation": equation_name,
            "description": equation_desc,
            "avg_iter": avg_iter,
            "behavior_score": behavior_score,
            "emergence": emergence,
            "fractal_dim": fractal_dim,
            "insights": [],
            "c_metrics": c_metrics,
            "seed_origin": f"Seed {seed_idx + 1} (Experiment {seed['experiment']})"  # Updated to use seed['experiment']
        })

        # Insights for summary
        if "exp(1j" in equation_name:
            experiment_results[-1]["insights"].append("Quantum effect: Phase terms may oscillate z values.")
        if "z/|z|" in equation_name:
            experiment_results[-1]["insights"].append("Quantum effect: Normalization may cap growth.")
        if avg_iter < 2.5:
            experiment_results[-1]["insights"].append("Insight: Classical divergence tempered by quantum terms.")
        elif avg_iter > 7:
            experiment_results[-1]["insights"].append("Insight: Quantum stability dominating classical chaos.")
        else:
            experiment_results[-1]["insights"].append("Insight: Balanced classical-quantum interplay.")

        # Print emergent findings for first 5 experiments
        if i <= 5 and "Emergent" in emergence:
            print(f"Emergent finding in Experiment {i+1}: {emergence}")

        # Progress update and checkpoint
        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1} experiments...")
        if (i + 1) % 1000 == 0:
            save_checkpoint(experiment_results, i + 1)
            print(f"Checkpoint saved at experiment {i + 1}.")

    # Update novelty scores with mean iteration
    mean_iter = mean([res["avg_iter"] for res in experiment_results]) if experiment_results else 0
    for res in experiment_results:
        exp_num = res["experiment"] - 1
        center_history = center_histories.get(exp_num, [])
        res["behavior_score"] = compute_novelty_score(center_history, res["avg_iter"], mean_iter, max_iter, res["fractal_dim"])

    # High-level summary
    avg_iters = [res["avg_iter"] for res in experiment_results]
    mean_iter = mean(avg_iters) if avg_iters else 0
    stable_experiments = sum(1 for res in experiment_results if res["avg_iter"] > 7)
    chaotic_experiments = sum(1 for res in experiment_results if res["avg_iter"] < 2.5)
    balanced_experiments = sum(1 for res in experiment_results if 2.5 <= res["avg_iter"] <= 7)
    emergent_experiments = sum(1 for res in experiment_results if "Emergent" in res["emergence"])

    print("\nHigh-Level Summary of 10,000 Experiments (Seeded):")
    print(f"Mean average iteration count: {mean_iter:.2f}")
    print(f"Stable experiments (>7 avg iters): {stable_experiments} ({stable_experiments/num_experiments*100:.1f}%)")
    print(f"Chaotic experiments (<2.5 avg iters): {chaotic_experiments} ({chaotic_experiments/num_experiments*100:.1f}%)")
    print(f"Balanced experiments (2.5-7 avg iters): {balanced_experiments} ({balanced_experiments/num_experiments*100:.1f}%)")
    print(f"Emergent experiments: {emergent_experiments} ({emergent_experiments/num_experiments*100:.1f}%)")
    print(f"Average fractal dimension: {mean([res['fractal_dim'] for res in experiment_results]):.2f}")

    # Top 10 novel formulas
    sorted_results = sorted(experiment_results, key=lambda x: x["behavior_score"], reverse=True)
    print("\nTop 10 Novel Formulas/Connections Found:")
    for i, res in enumerate(sorted_results[:10], 1):
        print(f"\n{i}. Experiment {res['experiment']}: z_(n+1) = {res['equation']}")
        print(f"Derived from: {res['seed_origin']}")
        print(f"Description: {res['description']}")
        print(f"Avg iteration count: {res['avg_iter']:.2f}, Novelty score: {res['behavior_score']:.2f}")
        print(f"Emergent behavior: {res['emergence']}")
        print(f"Fractal dimension: {res['fractal_dim']:.2f}")
        for insight in res["insights"]:
            print(insight)
        if "Emergent" in res["emergence"]:
            print("Potential novel math: This equation may define a new iterative system or connection.")
        if f"c={center_c:.2f}" in res["c_metrics"]:
            print(f"  Avg |z| at c=-0.50+0.00j: {res['c_metrics'][center_c]:.3f}")

    # Log summary
    logging.info("\nHigh-Level Summary of 10,000 Experiments (Seeded):")
    logging.info(f"Mean average iteration count: {mean_iter:.2f}")
    logging.info(f"Stable experiments (>7 avg iters): {stable_experiments} ({stable_experiments/num_experiments*100:.1f}%)")
    logging.info(f"Chaotic experiments (<2.5 avg iters): {chaotic_experiments} ({chaotic_experiments/num_experiments*100:.1f}%)")
    logging.info(f"Balanced experiments (2.5-7 avg iters): {balanced_experiments} ({balanced_experiments/num_experiments*100:.1f}%)")
    logging.info(f"Emergent experiments: {emergent_experiments} ({emergent_experiments/num_experiments*100:.1f}%)")
    logging.info(f"Average fractal dimension: {mean([res['fractal_dim'] for res in experiment_results]):.2f}")
    logging.info("\nTop 10 Novel Formulas/Connections Found:")
    for i, res in enumerate(sorted_results[:10], 1):
        logging.info(f"\n{i}. Experiment {res['experiment']}: z_(n+1) = {res['equation']}")
        logging.info(f"Derived from: {res['seed_origin']}")
        logging.info(f"Description: {res['description']}")
        logging.info(f"Avg iteration count: {res['avg_iter']:.2f}, Novelty score: {res['behavior_score']:.2f}")
        logging.info(f"Emergent behavior: {res['emergence']}")
        logging.info(f"Fractal dimension: {res['fractal_dim']:.2f}")
        for insight in res["insights"]:
            logging.info(insight)
        if "Emergent" in res["emergence"]:
            logging.info("Potential novel math: This equation may define a new iterative system or connection.")
        if f"c={center_c:.2f}" in res["c_metrics"]:
            logging.info(f"  Avg |z| at c=-0.50+0.00j: {res['c_metrics'][center_c]:.3f}")

    print(f"\nExploration complete. Detailed numbers in console (first 5 experiments) and log: {log_filename}")
    logging.info("\nExploration complete.")
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


import numpy as np
import logging
from datetime import datetime
import random
from statistics import mean, stdev
import os
import pickle

# Set up logging
log_filename = f"emergent_novel_math_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

# Checkpoint file to save progress
checkpoint_file = "checkpoint.pkl"

def save_checkpoint(experiment_results, current_exp):
    """Save progress to a checkpoint file."""
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'experiment_results': experiment_results, 'current_exp': current_exp}, f)

def load_checkpoint():
    """Load progress from a checkpoint file."""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        return data['experiment_results'], data['current_exp']
    return [], 0

def generate_fusion_equation():
    """Generate an equation blending classical and quantum math."""
    classical_terms = [
        ("z**2", lambda z, c: z**2, "Classical: Quadratic nonlinearity"),
        ("z**3", lambda z, c: z**3, "Classical: Cubic nonlinearity"),
        ("z", lambda z, c: z, "Classical: Linear term"),
        ("c", lambda z, c: c, "Classical: External parameter")
    ]
    quantum_terms = [
        ("exp(1j*|z|)*z", lambda z, c: np.exp(1j * abs(z)) * z, "Quantum: Phase rotation from magnitude"),
        ("exp(1j*c.real)*z", lambda z, c: np.exp(1j * c.real) * z, "Quantum: Phase from real part"),
        ("z/|z|", lambda z, c: z / max(abs(z), 0.01), "Quantum: Normalized state")
    ]
    
    c_terms = random.sample(classical_terms, random.randint(1, 2))
    q_terms = random.sample(quantum_terms, random.randint(1, 2))
    all_terms = c_terms + q_terms
    coeffs = [random.uniform(-1, 1) for _ in range(len(all_terms))]
    
    def equation(z, c):
        result = 0
        for (_, term_func, _), coeff in zip(all_terms, coeffs):
            result += coeff * term_func(z, c)
        if np.isnan(result) or np.isinf(result):
            return z # Return previous z if invalid
        return result
    
    equation_str = " + ".join(f"{coeff:.2f}*{term}" for (term, _, _), coeff in zip(all_terms, coeffs))
    equation_desc = "; ".join(f"{coeff:.2f}*{desc}" for (_, _, desc), coeff in zip(all_terms, coeffs))
    return equation, equation_str, equation_desc

def generate_fractal(width, height, x_min, x_max, y_min, y_max, max_iter, equation_func, equation_name, experiment_num):
    """Compute iterations and track numbers for sample points with initial z."""
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    fractal = np.zeros((height, width))
    
    # Exact sample points within range
    sample_cs = [
        complex(-0.5, 0.0), # Center
        complex(-2.0, 1.5), # Top-left
        complex(1.0, -1.5) # Bottom-right
    ]
    sample_histories = {f"c={c:.2f}": [] for c in sample_cs}

    # Print and log for first 5 experiments only
    if experiment_num <= 5:
        print(f"\nExperiment {experiment_num} for: z_(n+1) = {equation_name}")
        print(f"Description: {equation_desc}")
        logging.info(f"\nExperiment {experiment_num}: z_(n+1) = {equation_name}")
        logging.info(f"Description: {equation_desc}")

    for i in range(height):
        for j in range(width):
            c = x[j] + y[i]*1j
            z = complex(random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01))
            for k in range(max_iter):
                z_prev = z
                z = equation_func(z, c)
                if c in [s.real + s.imag*1j for s in sample_cs]:
                    c_key = f"c={c:.2f}"
                    sample_histories[c_key].append(z)
                    if experiment_num <= 5 and len(sample_histories[c_key]) <= 5:
                        print(f"{c_key}, Iter {k}: z = {z.real:.3f} + {z.imag:.3f}i, |z| = {abs(z):.3f}")
                        logging.info(f"{c_key}, Iter {k}: z = {z.real:.3f} + {z.imag:.3f}i, |z| = {abs(z):.3f}")
                if abs(z) > 2 or z == z_prev: # Detect stall
                    fractal[i, j] = k
                    break
            else:
                fractal[i, j] = max_iter
    
    return fractal, sample_histories

def log_numbers(history, equation_name, sample_c, experiment_num):
    """Log detailed iteration numbers only for first 5 experiments."""
    if experiment_num <= 5:
        logging.info(f"Sample point {sample_c} under {equation_name}:")
        for i, z in enumerate(history):
            logging.info(f"Iter {i}: z = {z.real:.3f} + {z.imag:.3f}i, |z| = {abs(z):.3f}")
        if len(history) < max_iter:
            logging.info(f"Diverged or stalled after {len(history)} iterations.")
        else:
            logging.info(f"Bounded up to {max_iter} iterations.")

def detect_emergence(history):
    """Detect emergent patterns in z evolution."""
    if len(history) < 3:
        return "No emergence detectable."
    magnitudes = [abs(z) for z in history]
    diffs = np.diff(magnitudes)
    # Check for near-cycles
    if len(diffs) > 2 and np.any(np.abs(diffs[1:] - diffs[:-1]) < 0.005):
        return "Emergent: Possible oscillation or near-cycle detected."
    # Check for slow convergence
    if len(diffs) > 5 and np.any(np.abs(diffs[-3:]) < 0.001):
        return "Emergent: Slow convergence to fixed point."
    # Check for fixed points
    if len(diffs) > 2 and all(abs(d) < 0.0001 for d in diffs):
        return "Emergent: Fixed point detected."
    # Check for amplitude shifts
    if max(magnitudes) - min(magnitudes) > 0.3 and len(history) > 3:
        return "Emergent: Significant amplitude shift observed."
    # Check for unexpected stability
    if all(m < 1.0 for m in magnitudes) and len(history) == max_iter:
        return "Emergent: Unexpected stability despite classical terms."
    return "No clear emergent pattern."

def compute_novelty_score(history, avg_iter, mean_iter, max_iter):
    """Score novelty based on deviation and emergent behavior."""
    if len(history) < 3:
        return 0
    magnitudes = [abs(z) for z in history]
    oscillation = stdev(magnitudes) if len(magnitudes) > 1 else 0
    stability = avg_iter / max_iter # Normalize to 0-1
    deviation = abs(avg_iter - mean_iter) / max_iter # Normalize deviation
    emergence_bonus = 0.6 if "Emergent" in detect_emergence(history) else 0
    term_diversity = len(magnitudes) / max_iter # Normalize by max iterations
    return 0.3 * oscillation + 0.2 * stability + 0.2 * deviation + 0.2 * term_diversity + emergence_bonus

# Main execution
if __name__ == "__main__":
    # Optimized settings for Pydroid 3
    width = 25 # Reduced to minimize memory usage
    height = 25
    x_min, x_max = -2.0, 1.0
    y_min, y_max = -1.5, 1.5
    max_iter = 10
    num_experiments = 10000

    print("Starting classical-quantum fusion for 10,000 experiments.")
    print(f"Grid: {width}x{height}, Max iters: {max_iter}, Range: Re [{x_min}, {x_max}], Im [{y_min}, {y_max}]")
    logging.info("Starting classical-quantum fusion for 10,000 experiments.")
    logging.info(f"Grid: {width}x{height}, Max iters: {max_iter}, Range: Re [{x_min}, {x_max}], Im [{y_min}, {y_max}]")

    # Load checkpoint if exists
    experiment_results, start_exp = load_checkpoint()
    if start_exp > 0:
        print(f"Resuming from experiment {start_exp}...")
    else:
        experiment_results = []

    center_c = complex(-0.5, 0.0) # Consistent center point
    center_histories = {} # Store only center point history to save memory

    for i in range(start_exp, num_experiments):
        # Generate equation
        try:
            equation_func, equation_name, equation_desc = generate_fusion_equation()
        except Exception as e:
            logging.error(f"Error generating equation for experiment {i+1}: {str(e)}")
            continue

        # Generate fractal and track numbers
        try:
            fractal, sample_histories = generate_fractal(width, height, x_min, x_max, y_min, y_max, 
                                                        max_iter, equation_func, equation_name, i+1)
        except (MemoryError, Exception) as e:
            logging.error(f"Error in experiment {i+1}: {str(e)}")
            continue

        # Log detailed numbers only for first 5 experiments
        if i <= 5:
            for sample_c, history in sample_histories.items():
                log_numbers(history, equation_name, sample_c, i+1)

        # Compute metrics
        avg_iter = np.mean(fractal)
        center_history = sample_histories.get(f"c={center_c:.2f}", [])
        center_histories[i] = center_history # Store only center history
        emergence = detect_emergence(center_history)
        behavior_score = compute_novelty_score(center_history, avg_iter, 0, max_iter)

        # Store results
        experiment_results.append({
            "experiment": i+1,
            "equation": equation_name,
            "description": equation_desc,
            "avg_iter": avg_iter,
            "behavior_score": behavior_score,
            "emergence": emergence,
            "insights": []
        })

        # Insights for summary
        if "exp(1j" in equation_name:
            experiment_results[-1]["insights"].append("Quantum effect: Phase terms may oscillate z values.")
        if "z/|z|" in equation_name:
            experiment_results[-1]["insights"].append("Quantum effect: Normalization may cap growth.")
        if avg_iter < 2.5:
            experiment_results[-1]["insights"].append("Insight: Classical divergence tempered by quantum terms.")
        elif avg_iter > 7:
            experiment_results[-1]["insights"].append("Insight: Quantum stability dominating classical chaos.")
        else:
            experiment_results[-1]["insights"].append("Insight: Balanced classical-quantum interplay.")

        # Print emergent findings for first 5 experiments
        if i <= 5 and "Emergent" in emergence:
            print(f"Emergent finding in Experiment {i+1}: {emergence}")

        # Progress update and checkpoint
        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1} experiments...")
        if (i + 1) % 1000 == 0:
            save_checkpoint(experiment_results, i + 1)
            print(f"Checkpoint saved at experiment {i + 1}.")

    # Update novelty scores with mean iteration
    mean_iter = mean([res["avg_iter"] for res in experiment_results]) if experiment_results else 0
    for res in experiment_results:
        exp_num = res["experiment"] - 1
        center_history = center_histories.get(exp_num, [])
        res["behavior_score"] = compute_novelty_score(center_history, res["avg_iter"], mean_iter, max_iter)

    # High-level summary
    avg_iters = [res["avg_iter"] for res in experiment_results]
    mean_iter = mean(avg_iters) if avg_iters else 0
    stable_experiments = sum(1 for res in experiment_results if res["avg_iter"] > 7)
    chaotic_experiments = sum(1 for res in experiment_results if res["avg_iter"] < 2.5)
    balanced_experiments = sum(1 for res in experiment_results if 2.5 <= res["avg_iter"] <= 7)
    emergent_experiments = sum(1 for res in experiment_results if "Emergent" in res["emergence"])

    print("\nHigh-Level Summary of 10,000 Experiments:")
    print(f"Mean average iteration count: {mean_iter:.2f}")
    print(f"Stable experiments (>7 avg iters): {stable_experiments} ({stable_experiments/num_experiments*100:.1f}%)")
    print(f"Chaotic experiments (<2.5 avg iters): {chaotic_experiments} ({chaotic_experiments/num_experiments*100:.1f}%)")
    print(f"Balanced experiments (2.5-7 avg iters): {balanced_experiments} ({balanced_experiments/num_experiments*100:.1f}%)")
    print(f"Emergent experiments: {emergent_experiments} ({emergent_experiments/num_experiments*100:.1f}%)")

    # Top 10 novel formulas
    sorted_results = sorted(experiment_results, key=lambda x: x["behavior_score"], reverse=True)
    print("\nTop 10 Novel Formulas/Connections Found:")
    for i, res in enumerate(sorted_results[:10], 1):
        print(f"\n{i}. Experiment {res['experiment']}: z_(n+1) = {res['equation']}")
        print(f"Description: {res['description']}")
        print(f"Avg iteration count: {res['avg_iter']:.2f}, Novelty score: {res['behavior_score']:.2f}")
        print(f"Emergent behavior: {res['emergence']}")
        for insight in res["insights"]:
            print(insight)
        if "Emergent" in res["emergence"]:
            print("Potential novel math: This equation may define a new iterative system or connection.")

    # Log summary
    logging.info("\nHigh-Level Summary of 10,000 Experiments:")
    logging.info(f"Mean average iteration count: {mean_iter:.2f}")
    logging.info(f"Stable experiments (>7 avg iters): {stable_experiments} ({stable_experiments/num_experiments*100:.1f}%)")
    logging.info(f"Chaotic experiments (<2.5 avg iters): {chaotic_experiments} ({chaotic_experiments/num_experiments*100:.1f}%)")
    logging.info(f"Balanced experiments (2.5-7 avg iters): {balanced_experiments} ({balanced_experiments/num_experiments*100:.1f}%)")
    logging.info(f"Emergent experiments: {emergent_experiments} ({emergent_experiments/num_experiments*100:.1f}%)")
    logging.info("\nTop 10 Novel Formulas/Connections Found:")
    for i, res in enumerate(sorted_results[:10], 1):
        logging.info(f"\n{i}. Experiment {res['experiment']}: z_(n+1) = {res['equation']}")
        logging.info(f"Description: {res['description']}")
        logging.info(f"Avg iteration count: {res['avg_iter']:.2f}, Novelty score: {res['behavior_score']:.2f}")
        logging.info(f"Emergent behavior: {res['emergence']}")
        for insight in res["insights"]:
            logging.info(insight)
        if "Emergent" in res["emergence"]:
            logging.info("Potential novel math: This equation may define a new iterative system or connection.")

    print(f"\nExploration complete. Detailed numbers in console (first 5 experiments) and log: {log_filename}")
    logging.info("\nExploration complete.")
    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
