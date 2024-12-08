# MiKayla Carney
# Dr. Yang Li
# MATH: 3020-008
# 12/12/2024

import numpy as np

def monte_carlo_integration_part_a(n_samples=100000):
    # Function to integrate
    def f(x):
        return np.sin(1 / x)

    # Generate random samples uniformly in (0, 1)
    x = np.random.uniform(0, 1, n_samples)
    
    # Evaluate function, handling division by zero near zero
    fx = np.sin(1 / np.clip(x, 1e-10, None))

    # Monte Carlo integration
    integral = (1 - 0) * np.mean(fx)
    return integral


def monte_carlo_integration_part_b(n_samples=100000):
    # Function to integrate
    def g(x):
        return np.exp(-x**2)

    # Generate random samples uniformly in (-2, 2)
    x = np.random.uniform(-2, 2, n_samples)

    # Evaluate the function
    gx = g(x)

    # Monte Carlo integration
    integral = (2 - (-2)) * np.mean(gx)
    return integral


# Simulation for Virus Spread in Network
def virus_spread_simulation(n_simulations=1000):
    n_computers = 20
    spread_prob = 0.1
    removal_per_day = 5

    times_to_clean = []
    all_infected_counts = []
    total_infected_counts = []

    for _ in range(n_simulations):
        infected = np.zeros(n_computers, dtype=bool)
        infected[0] = True  # Start with one infected computer
        days = 0

        infected_once = np.zeros(n_computers, dtype=bool)

        while np.any(infected):
            days += 1
            infected_once = np.logical_or(infected_once, infected)

            # Spread the virus
            for i in range(n_computers):
                if infected[i]:
                    for j in range(n_computers):
                        if not infected[j] and np.random.random() < spread_prob:
                            infected[j] = True

            # Remove the virus from up to 5 infected computers
            infected_indices = np.where(infected)[0]
            if len(infected_indices) > 0:
                to_remove = np.random.choice(infected_indices, min(len(infected_indices), removal_per_day), replace=False)
                infected[to_remove] = False

        times_to_clean.append(days)
        all_infected_counts.append(np.all(infected_once))
        total_infected_counts.append(np.sum(infected_once))

    expected_time = np.mean(times_to_clean)
    prob_all_infected = np.mean(all_infected_counts)
    expected_total_infected = np.mean(total_infected_counts)

    return expected_time, prob_all_infected, expected_total_infected

# Main execution
if __name__ == "__main__":
    # Question 1
    integral_a = monte_carlo_integration_part_a()
    integral_b = monte_carlo_integration_part_b()

    print(f"Monte Carlo estimate of integral (a): {integral_a}")
    print(f"Monte Carlo estimate of integral (b): {integral_b}")

    # Question 2
    expected_time, prob_all_infected, expected_total_infected = virus_spread_simulation()

    print(f"Expected time to remove the virus: {expected_time:.2f} days")
    print(f"Probability that all computers are infected at least once: {prob_all_infected:.4f}")
    print(f"Expected number of computers infected: {expected_total_infected:.2f}")
