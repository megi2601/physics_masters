import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from dwave.samplers import SimulatedAnnealingSampler
import pandas as pd
import itertools
import numpy as np
import time
import os

###############################################################################################
# ------------------------- Build QUBO -------------------------
def build_qubo(N, lam=1.0):
    def linear_index(i, j, k):
        return i * N * 2 + j * 2 + k
    Q = {}
    coeffs = {
        (0, 0):  1, (1, 1):  1, (2, 2):  1, (3, 3):  1,
        (0, 1): -2, (0, 2): -2, (0, 3):  2,
        (1, 2):  2, (1, 3): -2, (2, 3): -2
    }
    for i, j in itertools.product(range(N), range(N)):
        coords = [
            (i, j, 0),
            (i, j, 1),
            ((i - 1) % N, j, 0),
            (i, (j - 1) % N, 1),
        ]
        idx = [linear_index(*c) for c in coords]
        for (a, b), w in coeffs.items():
            u, v = idx[a], idx[b]
            if u > v:
                u, v = v, u
            Q[(u, v)] = Q.get((u, v), 0.0) + lam * w
    return Q

###############################################################################################
# ------------------------- Solve BQM -------------------------
def solve_bqm(Q, sampler, params=dict(), worker_id=0):
    num_reads = params.pop('num_reads', 40)
    chunk_size = 20
    all_samples, all_energies = [], []

    for start in tqdm(range(0, num_reads, chunk_size), desc=f"[Worker {worker_id}] sampling", position=worker_id):
        chunk_params = dict(params)
        chunk_params['num_reads'] = min(chunk_size, num_reads - start)
        sampleset = sampler.sample_qubo(Q, **chunk_params)
        samples = np.array([list(s.values()) for s in sampleset.samples()])
        energies = sampleset.record.energy
        all_samples.append(samples)
        all_energies.append(energies)

    samples = np.vstack(all_samples)
    energies = np.concatenate(all_energies)
    return samples, energies

###############################################################################################
# ------------------------- Pattern check -------------------------
def stN(solution):
    N = int(np.sqrt(len(solution) / 2))
    def linear_index(i, j, k):
        return i * N * 2 + j * 2 + k
    sol = np.asarray(solution).astype(int)
    patterns = {
        "0000": np.array([0, 0, 0, 0]),
        "1111": np.array([1, 1, 1, 1]),
        "0011": np.array([0, 0, 1, 1]),
        "1100": np.array([1, 1, 0, 0]),
    }
    counts = {name: 0 for name in patterns}
    for i, j in itertools.product(range(N), range(N)):
        coords = [
            (i, j, 0),
            (i, j, 1),
            ((i - 1) % N, j, 0),
            (i, (j - 1) % N, 1),
        ]
        idx = [linear_index(*c) for c in coords]
        values = sol[idx]
        for name, pattern in patterns.items():
            if np.array_equal(values, pattern):
                counts[name] += 1
                break
    total_nodes = N * N
    total = sum(counts.values()) / total_nodes
    return total

###############################################################################################
# ------------------------- Worker function -------------------------
def worker_run(worker_id, N, Q, beta_start, beta_end, num_reads_each):
    np.random.seed(int(time.time() * 1000) % (2**32 - 1))
    start_time = time.time()
    print(f"[Worker {worker_id}] Starting simulation with {num_reads_each} reads")

    sampler = SimulatedAnnealingSampler()
    params = {
        'num_reads': num_reads_each,
        'randomize_order': True,
        'beta_range': [beta_start, beta_end],
        'num_sweeps': (N // 16) * 2 * 1000
    }

    samples, energies = solve_bqm(Q, sampler, params, worker_id)
    stNs = np.array([stN(sample) for sample in samples])

    df = pd.DataFrame({'E': energies, 'stN': stNs})
    fname = f'tmp_worker_{worker_id}.csv'
    df.to_csv(fname, index=False)
    print(f"[Worker {worker_id}] Finished in {(time.time()-start_time):.1f}s, saved {fname}")
    return fname

###############################################################################################
# ------------------------- Main -------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn")  # safer cross-platform
    N = 256
    beta_start, beta_end = 10, 1000
    total_reads = 240
    num_workers = 6
    reads_per_worker = total_reads // num_workers

    print(f"Building QUBO for N={N} ...")
    Q = build_qubo(N)

    print(f"Distributing {total_reads} reads across {num_workers} workers "
          f"({reads_per_worker} each)...")

    start = time.time()
    with mp.Pool(processes=num_workers) as pool:
        worker_func = partial(worker_run, N=N, Q=Q,
                              beta_start=beta_start,
                              beta_end=beta_end,
                              num_reads_each=reads_per_worker)
        results = list(pool.imap_unordered(worker_func, range(num_workers)))

    # Combine all CSVs
    dfs = [pd.read_csv(f) for f in results]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(f'simulated_annealing_N{N}_combined.csv', index=False)

    # Cleanup temp files
    for f in results:
        os.remove(f)

    print(f"\n✅ Finished all workers in {(time.time()-start)/60:.2f} min")
    print(f"Results saved to simulated_annealing_N{N}_combined.csv")
