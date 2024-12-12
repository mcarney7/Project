# MiKayla Carney
# Dr. Yang Li
# MATH: 3020-008
# 12/12/2024

import numpy as np

# Monte Carlo integration to estimate \int_0^1 sin(1/x) dx
def monte_carlo_integration_part_a(n_samples=100000):
    """
    Estimate the integral \int_0^1 sin(1/x) dx using Monte Carlo integration.
    n_samples: Number of random samples to generate for the estimation.
    """
    # Generate random samples between 0 and 1
    x = np.random.uniform(0, 1, n_samples)
    # Calculate sin(1/x) for each sample, avoiding division by zero
    fx = np.sin(1 / np.clip(x, 1e-10, None))
    # Estimate the integral as the average value of the function scaled by the interval length
    integral = (1 - 0) * np.mean(fx)
    return integral

# Monte Carlo integration to estimate \int_{-2}^2 e^(-x^2) dx
def monte_carlo_integration_part_b(n_samples=100000):
    """
    Estimate the integral \int_{-2}^2 e^(-x^2) dx using Monte Carlo integration.
    n_samples: Number of random samples to generate for the estimation.
    """
    # Generate random samples between -2 and 2
    x = np.random.uniform(-2, 2, n_samples)
    # Calculate e^(-x^2) for each sample
    fx = np.exp(-x**2)
    # Estimate the integral as the average value of the function scaled by the interval length
    integral = (2 - (-2)) * np.mean(fx)
    return integral

# Simulation to estimate the expected time to remove the virus from a network
def simplified_virus_clean_time_simulation(n_simulations=10):
    """
    Simulate the expected time to remove the virus from the network.
    n_simulations: Number of simulations to perform.
    """
    n_computers = 20  # Total number of computers in the network
    spread_prob = 0.1  # Probability of the virus spreading each day
    removal_per_day = 5  # Number of infected computers removed per day

    times_to_clean = []  # List to store time taken in each simulation

    for _ in range(n_simulations):
        # Initialize infection status; only the first computer is initially infected
        infected = np.zeros(n_computers, dtype=bool)
        infected[0] = True
        days = 0  # Counter for the number of days

        while np.any(infected):
            days += 1  # Increment the day counter

            # Simulate virus spread to other computers
            new_infections = np.random.random(n_computers) < spread_prob
            infected = np.logical_or(infected, new_infections)

            # Remove the virus from up to 5 infected computers
            infected_indices = np.where(infected)[0]
            if len(infected_indices) > 0:
                to_remove = np.random.choice(infected_indices, min(len(infected_indices), removal_per_day), replace=False)
                infected[to_remove] = False

        times_to_clean.append(days)  # Record the time taken for this simulation

    # Calculate the expected time as the average over all simulations
    expected_time = np.mean(times_to_clean)
    return expected_time

# Simulation to estimate the probability that all computers are infected at least once
def simplified_virus_all_infected_simulation(n_simulations=10):
    """
    Simulate the probability that all computers are infected at least once.
    n_simulations: Number of simulations to perform.
    """
    n_computers = 20  # Total number of computers in the network
    spread_prob = 0.1  # Probability of the virus spreading each day
    removal_per_day = 5  # Number of infected computers removed per day

    all_infected_counts = []  # List to store whether all computers were infected in each simulation

    for _ in range(n_simulations):
        # Initialize infection status; only the first computer is initially infected
        infected = np.zeros(n_computers, dtype=bool)
        infected[0] = True

        # Track computers that have been infected at least once
        infected_once = np.zeros(n_computers, dtype=bool)

        while np.any(infected):
            # Update the list of computers that have been infected at least once
            infected_once = np.logical_or(infected_once, infected)

            # Simulate virus spread to other computers
            new_infections = np.random.random(n_computers) < spread_prob
            infected = np.logical_or(infected, new_infections)

            # Remove the virus from up to 5 infected computers
            infected_indices = np.where(infected)[0]
            if len(infected_indices) > 0:
                to_remove = np.random.choice(infected_indices, min(len(infected_indices), removal_per_day), replace=False)
                infected[to_remove] = False

        # Check if all computers were infected at least once
        all_infected_counts.append(np.all(infected_once))

    # Calculate the probability as the average over all simulations
    prob_all_infected = np.mean(all_infected_counts)
    return prob_all_infected

# Simulation to estimate the expected number of computers that get infected at least once
def simplified_virus_total_infected_simulation(n_simulations=10):
    """
    Simulate the expected number of computers that get infected at least once.
    n_simulations: Number of simulations to perform.
    """
    n_computers = 20  # Total number of computers in the network
    spread_prob = 0.1  # Probability of the virus spreading each day
    removal_per_day = 5  # Number of infected computers removed per day

    total_infected_counts = []  # List to store the number of computers infected in each simulation

    for _ in range(n_simulations):
        # Initialize infection status; only the first computer is initially infected
        infected = np.zeros(n_computers, dtype=bool)
        infected[0] = True

        # Track computers that have been infected at least once
        infected_once = np.zeros(n_computers, dtype=bool)

        while np.any(infected):
            # Update the list of computers that have been infected at least once
            infected_once = np.logical_or(infected_once, infected)

            # Simulate virus spread to other computers
            new_infections = np.random.random(n_computers) < spread_prob
            infected = np.logical_or(infected, new_infections)

            # Remove the virus from up to 5 infected computers
            infected_indices = np.where(infected)[0]
            if len(infected_indices) > 0:
                to_remove = np.random.choice(infected_indices, min(len(infected_indices), removal_per_day), replace=False)
                infected[to_remove] = False

        # Count the number of computers that were infected at least once
        total_infected_counts.append(np.sum(infected_once))

    # Calculate the expected number of infected computers as the average over all simulations
    expected_total_infected = np.mean(total_infected_counts)
    return expected_total_infected

# Main execution
if __name__ == "__main__":
    # Question 1 (a): Estimate \int_0^1 sin(1/x) dx
    integral_a = monte_carlo_integration_part_a()
    print(f"Estimated value of integral (a): {integral_a:.4f}")

    # Question 1 (b): Estimate \int_{-2}^2 e^(-x^2) dx
    integral_b = monte_carlo_integration_part_b()
    print(f"Estimated value of integral (b): {integral_b:.4f}")

    # Question 2 (a): Estimate the expected time to remove the virus
    expected_time = simplified_virus_clean_time_simulation(n_simulations=1000)
    print(f"Expected time to remove the virus from the entire network: {expected_time:.1f} days")

    # Question 2 (b): Estimate the probability that all computers get infected at least once
    prob_all_infected = simplified_virus_all_infected_simulation(n_simulations=1000)
    print(f"Probability that all computers get infected at least once: {prob_all_infected:.3f}")

    # Question 2 (c): Estimate the expected number of computers that get infected
    expected_total_infected = simplified_virus_total_infected_simulation(n_simulations=1000)
    print(f"Expected number of computers that get infected: {expected_total_infected:.1f}")
