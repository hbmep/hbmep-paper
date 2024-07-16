import numpy as np
import scipy.stats as stats


def power_analysis(pre, post, candidate_test='t-test', alpha=0.05, power_target=0.80, max_n=100, simulations=2500):
    differences = np.array(post) - np.array(pre)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    def calculate_power(n, test_type):
        power_count = 0
        for _ in range(simulations):
            sample = np.random.normal(mean_diff, std_diff, n)
            if test_type == 't-test':
                t_stat, p_value = stats.ttest_1samp(sample, 0)
            elif test_type == 'signrank':
                t_stat, p_value = stats.wilcoxon(sample)
            else:
                raise ValueError("Unsupported test type. Use 't-test' or 'signrank'.")
            if p_value < alpha:
                power_count += 1
        return power_count / simulations

    results = []
    for n in range(3, max_n + 1):
        power = calculate_power(n, candidate_test)
        results.append((n, power))
        if power >= power_target:
            break

    return results

# rng62, rng40, rng20
rng62_apb = 0.0
pre = np.array([29.0, 36.0, 39.0])
post = np.array([16.0+36.0+rng62_apb, 16.0+39.0+17.0, 14.0+28.0+15.0])  # not yet in
results_ttest = power_analysis(pre, post, candidate_test='t-test')
results_signrank = power_analysis(pre, post, candidate_test='signrank')

import matplotlib.pyplot as plt


def plot_power_analysis(results, title):
    ns, powers = zip(*results)
    plt.figure()
    plt.plot(ns, powers, marker='o')
    plt.axhline(y=0.80, color='r', linestyle='--')
    plt.xlabel('Sample Size (N)')
    plt.ylabel('Power')
    plt.title(title)
    plt.show()


# Plot results
plot_power_analysis(results_ttest, 'Power Analysis (Paired t-test)')
plot_power_analysis(results_signrank, 'Power Analysis (Wilcoxon Signed-Rank Test)')