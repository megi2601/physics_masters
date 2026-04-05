from dwave.samplers import SimulatedAnnealingSampler
import pandas as pd
import itertools
import numpy as np
import multiprocessing as mp
from functools import partial
from tqdm import tqdm  # for progress bar

# pip install dwave-ocean-sdk pandas numpy dimod

###############################################################################################################3

def build_qubo(N, lam=1.0):
    """
    Build QUBO enforcing (x[i,j,0] + x[i,(j-1)%N,1] - x[i,j,1] - x[(i-1)%N,j,0])^2 for all (i,j).

    Parameters
    ----------
    N : int
        Grid size.
    linear_index : function
        Maps (i,j,k) -> linear index in [0, N*N*2)
    lam : float
        Penalty weight (scales constraint).

    Returns
    -------
    dict
        QUBO dictionary {(u,v): coeff}
    """
    def linear_index(i, j, k):
        return i * N * 2 + j * 2 + k
    Q = {}

    # QUBO coefficients for (x1 + x4 - x2 - x3)^2
    coeffs = {
        (0, 0):  1, (1, 1):  1, (2, 2):  1, (3, 3):  1,
        (0, 1): -2, (0, 2): -2, (0, 3):  2,
        (1, 2):  2, (1, 3): -2, (2, 3): -2
    }

    for i, j in itertools.product(range(N), range(N)):
        # x1,x2,x3,x4 coordinates for this (i,j)
        coords = [
            (i, j, 0),              # x1
            (i, j, 1),              # x2
            ((i - 1) % N, j, 0),    # x3
            (i, (j - 1) % N, 1),    # x4
        ]
        # map to linear indices
        idx = [linear_index(*c) for c in coords]

        # accumulate scaled coefficients
        for (a, b), w in coeffs.items():
            u, v = idx[a], idx[b]
            if u > v:
                u, v = v, u  # keep upper-triangular for D-Wave
            Q[(u, v)] = Q.get((u, v), 0.0) + lam * w

    return Q

def solve_bqm(Q, sampler, params=dict()):

    sampleset = sampler.sample_qubo(Q, **params)
    
    samples = np.array([list(s.values()) for s in sampleset.samples()])
    energies = sampleset.record.energy

    return samples, energies


def stN(solution):
    """
    Given a flat binary solution vector and grid size N,
    compute the % of nodes (i,j) whose 4-bit pattern (v_ijk)
    is 1111, 0000, 1010, or 0101.

    Parameters
    ----------
    solution : array-like of {0,1}, length 4*N*N
    N : int, grid size
    """
    N = int(np.sqrt(len(solution) / 2))

    def linear_index(i, j, k):
        return i * N * 2 + j * 2 + k
        
    sol = np.asarray(solution).astype(int)
    assert sol.size == 2 * N * N, f"Expected {2*N*N} entries, got {sol.size}"


    # define patterns
    patterns = {
        "0000": np.array([0, 0, 0, 0]),
        "1111": np.array([1, 1, 1, 1]),
        "0011": np.array([0, 0 ,1,1]),
        "1100": np.array([1, 1, 0, 0]),
        # "1010": np.array([1, 0, 1, 0]),
        # "0101": np.array([0, 1, 0, 1]),
    }

    counts = {name: 0 for name in patterns}


    for i, j in itertools.product(range(N), range(N)):
        # x1,x2,x3,x4 coordinates for this (i,j)
        coords = [
            (i, j, 0),              # x1
            (i, j, 1),              # x2
            ((i - 1) % N, j, 0),    # x3
            (i, (j - 1) % N, 1),    # x4
        ]
        # map to linear indices
        idx = [linear_index(*c) for c in coords]

        values = sol[idx]
        for name, pattern in patterns.items():
            if np.array_equal(values, pattern):
                counts[name] += 1
                break

    total_nodes = N * N
    percentages = {name: counts[name] / total_nodes for name in counts}
    total = sum(counts.values()) / total_nodes
    return total

#############################################################################################

def run_simulation(i, N, beta_start, beta_end):
    n = N[i]
    sampler = SimulatedAnnealingSampler()
    Q = build_qubo(n)

    params = {
        'num_reads': 200,
        'randomize_order': True,
        'beta_range': [beta_start, beta_end],
        'num_sweeps': (n // 16) * 2 * 2**(i-2) * 1000
    }

    print(f"[Process {i}] Starting N={n}...")
    samples, energies = solve_bqm(Q, sampler, params)
    stNs = np.array([stN(sample) for sample in samples])

    df = pd.DataFrame({
        'final_configuration': ["".join(s.astype(str)) for s in samples],
        'E': energies,
        'stN': stNs
    })
    df[['E', 'stN']].to_csv(f'simulated_annealing_N{n}_summary_short.csv', index=False)
    df.to_csv(f'simulated_annealing_N{n}.csv', index=False)
    print(f"[Process {i}] Finished N={n}")




#########################################33
if __name__ == "__main__":
    N = np.array([16, 32, 64, 128, 256, 512])
    beta_start, beta_end = 10, 1000

    indices = [4]  # your selection
    with mp.Pool(processes=6) as pool:  # Use all 6 cores
        list(tqdm(pool.imap_unordered(
            partial(run_simulation, N=N, beta_start=beta_start, beta_end=beta_end),
            indices
        ), total=len(indices)))

