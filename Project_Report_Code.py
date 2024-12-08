# MiKayla Carney
# Dr. Yang Li
# MATH: 3020-008
# 12/12/2024

import numpy as np

def monte_carlo_integration_part_a(n_samples=100000):
    """
    Estimate the integral \int_0^1 sin(1/x) dx using Monte Carlo integration.
    """
    x = np.random.uniform(0, 1, n_samples)
    fx = np.sin(1 / np.clip(x, 1e-10, None))  # Avoid division by zero
    integral = (1 - 0) * np.mean(fx)
    return integral

def monte_carlo_integration_part_b(n_samples=100000):
    """
    Estimate the integral \int_{-2}^2 e^(-x^2) dx using Monte Carlo integration.
    """
    x = np.random.uniform(-2, 2, n_samples)
    fx = np.exp(-x**2)
    integral = (2 - (-2)) * np.mean(fx)
    return integral

def simplified_virus_clean_time_simulation(n_simulations=10):
    """
    Simulate the expected time to remove the virus from the network.
    """
    n_computers = 20
    spread_prob = 0.1
    removal_per_day = 5

    times_to_clean = []

    for _ in range(n_simulations):
        infected = np.zeros(n_computers, dtype=bool)
        infected[0] = True  # Start with one infected computer
        days = 0

        while np.any(infected):
            days += 1

            # Simplified virus spread
            new_infections = np.random.random(n_computers) < spread_prob
            infected = np.logical_or(infected, new_infections)

            # Remove the virus from up to 5 infected computers
            infected_indices = np.where(infected)[0]
            if len(infected_indices) > 0:
                to_remove = np.random.choice(infected_indices, min(len(infected_indices), removal_per_day), replace=False)
                infected[to_remove] = False

        times_to_clean.append(days)

    expected_time = np.mean(times_to_clean)
    return expected_time

def simplified_virus_all_infected_simulation(n_simulations=10):
    """
    Simulate the probability that all computers are infected at least once.
    """
    n_computers = 20
    spread_prob = 0.1
    removal_per_day = 5

    all_infected_counts = []

    for _ in range(n_simulations):
        infected = np.zeros(n_computers, dtype=bool)
        infected[0] = True  # Start with one infected computer

        infected_once = np.zeros(n_computers, dtype=bool)

        while np.any(infected):
            # Track which computers have ever been infected
            infected_once = np.logical_or(infected_once, infected)

            # Simplified virus spread
            new_infections = np.random.random(n_computers) < spread_prob
            infected = np.logical_or(infected, new_infections)

            # Remove the virus from up to 5 infected computers
            infected_indices = np.where(infected)[0]
            if len(infected_indices) > 0:
                to_remove = np.random.choice(infected_indices, min(len(infected_indices), removal_per_day), replace=False)
                infected[to_remove] = False

        all_infected_counts.append(np.all(infected_once))

    prob_all_infected = np.mean(all_infected_counts)
    return prob_all_infected

def simplified_virus_total_infected_simulation(n_simulations=10):
    """
    Simulate the expected number of computers that get infected at least once.
    """
    n_computers = 20
    spread_prob = 0.1
    removal_per_day = 5

    total_infected_counts = []

    for _ in range(n_simulations):
        infected = np.zeros(n_computers, dtype=bool)
        infected[0] = True  # Start with one infected computer

        infected_once = np.zeros(n_computers, dtype=bool)

        while np.any(infected):
            # Track which computers have ever been infected
            infected_once = np.logical_or(infected_once, infected)

            # Simplified virus spread
            new_infections = np.random.random(n_computers) < spread_prob
            infected = np.logical_or(infected, new_infections)

            # Remove the virus from up to 5 infected computers
            infected_indices = np.where(infected)[0]
            if len(infected_indices) > 0:
                to_remove = np.random.choice(infected_indices, min(len(infected_indices), removal_per_day), replace=False)
                infected[to_remove] = False

        total_infected_counts.append(np.sum(infected_once))

    expected_total_infected = np.mean(total_infected_counts)
    return expected_total_infected

# Main execution
if __name__ == "__main__":
    # Question 1 (a)
    integral_a = monte_carlo_integration_part_a()
    print(f"Estimated value of integral (a): {integral_a:.4f}")

    # Question 1 (b)
    integral_b = monte_carlo_integration_part_b()
    print(f"Estimated value of integral (b): {integral_b:.4f}")

    # Question 2 (a)
    expected_time = simplified_virus_clean_time_simulation(n_simulations=1000)
    print(f"Expected time to remove the virus from the entire network: {expected_time:.1f} days")

    # Question 2 (b)
    prob_all_infected = simplified_virus_all_infected_simulation(n_simulations=1000)
    print(f"Probability that all computers get infected at least once: {prob_all_infected:.3f}")

    # Question 2 (c)
    expected_total_infected = simplified_virus_total_infected_simulation(n_simulations=1000)
    print(f"Expected number of computers that get infected: {expected_total_infected:.1f}")
