import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Proof-of-Queue simulation suite with file output for both recovery and TPS plots

def simulate_queue(N,
                   adv_fraction,
                   strategy,     # 'self', 'random', 'withhold', 'invalid', 'fake'
                   p_miss=0.1,
                   publish_delay=5,
                   block_timeout=600,
                   max_turns=None,
                   seed=None,
                   debug=False):
    """
    Simulate PoQ with time accounting and optional fixed-duration.
    Returns: honest_blocks, fork_count, timeout_count, invalid_count,
             effective_tps, turns, malicious_remaining
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    initial_mal = int(N * adv_fraction)
    queue = deque(['M'] * initial_mal + ['H'] * (N - initial_mal))
    random.shuffle(queue)
    malicious_remaining = initial_mal

    honest_blocks = 0
    fork_count = timeout_count = invalid_count = 0
    ejected_since_last_honest = 0
    total_time = 0.0
    turns = 0

    while True:
        if max_turns is not None and turns >= max_turns:
            break
        if max_turns is None and malicious_remaining == 0:
            break
        if not queue:
            break

        publisher = queue.popleft()
        turns += 1

        if random.random() < p_miss:
            timeout_count += 1
            fork_count += 1
            total_time += block_timeout
            if publisher == 'M':
                malicious_remaining -= 1
            ejected_since_last_honest += 1
            continue

        total_time += publish_delay
        if publisher == 'H':
            honest_blocks += 1
            queue.extend(['H'] * (1 + ejected_since_last_honest))
            ejected_since_last_honest = 0
        else:
            if strategy == 'withhold':
                fork_count += 1
            elif strategy in ('invalid', 'fake'):
                invalid_count += 1
            malicious_remaining -= 1
            ejected_since_last_honest += 1

    effective_tps = honest_blocks / total_time if total_time > 0 else 0
    return honest_blocks, fork_count, timeout_count, invalid_count, effective_tps, turns, malicious_remaining


def evaluate_poq(recovery_mode=True, max_turns=10000):
    """Evaluate PoQ and save plots for recovery and TPS."""
    N_list = [200]
    adv_fracs = np.arange(0.1,1.0,0.1)
    p_miss_list = [0.05,0.1,0.2]
    strategies = ['self','random','withhold','invalid','fake']
    n_runs = 50

    for N in N_list:
        for p_miss in p_miss_list:
            # collect stats per (adv_frac, strategy)
            stats = {(f, strat): None for f in adv_fracs for strat in strategies}
            for f in adv_fracs:
                for strat in strategies:
                    runs = [simulate_queue(N,f,strat,
                                            p_miss=p_miss,
                                            publish_delay=5,
                                            block_timeout=600,
                                            max_turns=(None if recovery_mode else max_turns),
                                            seed=i)
                            for i in range(n_runs)]
                    df = pd.DataFrame(runs,
                                      columns=['honest','forks','timeouts','invalids','tps','turns','mal_rem'])
                    metric = 'turns' if recovery_mode else 'tps'
                    mean_val = df[metric].mean()
                    ci_val = 1.96 * df[metric].std() / np.sqrt(n_runs)
                    stats[(f,strat)] = (mean_val, ci_val)

            # save recovery or TPS plot
            title = 'Recovery Turns' if recovery_mode else 'Effective TPS'
            ylabel = 'Turns ±95% CI' if recovery_mode else 'TPS ±95% CI'
            plt.figure(figsize=(6,4))
            for strat in strategies:
                xs = []
                ys = []
                yerrs = []
                for f in adv_fracs:
                    mean_val, ci_val = stats[(f,strat)]
                    xs.append(f); ys.append(mean_val); yerrs.append(ci_val)
                plt.errorbar(xs, ys, yerr=yerrs, marker='o')
            plt.title(f"{title} (N={N}, p_miss={p_miss})")
            plt.xlabel("Adversary Fraction")
            plt.ylabel(ylabel)
            plt.tight_layout()
            fname = f"{title.replace(' ','_')}_N{N}_p{int(p_miss*100)}.png"
            plt.savefig(fname)
            plt.close()

if __name__ == '__main__':
    evaluate_poq(recovery_mode=True)
    evaluate_poq(recovery_mode=False)
