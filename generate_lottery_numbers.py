from collections import Counter
from scipy.stats import dirichlet
import numpy as np

def read_data(file_path):
    """
    Reads the lottery data from the file and structures it into a 2D numpy array.
    
    Each line in the file contains a series of numbers without delimiters.
    The function splits these numbers into groups of two digits each.

    Args:
    file_path (str): Path to the file containing lottery data.

    Returns:
    np.array: A 2D numpy array where each row represents a set of lottery numbers.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    structured_data = []
    for line in lines:
        numbers_str = line.strip()
        numbers_list = [int(numbers_str[i:i+2]) for i in range(0, len(numbers_str), 2)]
        structured_data.append(numbers_list)
    return np.array(structured_data)

def estimate_hierarchical_probabilities(numbers, alpha=1, years=None, days=None, time_decay_factor=0.9):
    """
    Estimates the probabilities of drawing certain numbers using a hierarchical Bayesian method.
    
    If provided, uses years and days data to apply a time decay factor to give more weight to recent data.

    Args:
    numbers (np.array): Flattened array of lottery numbers.
    alpha (float): Dirichlet prior parameter.
    years (np.array): Array representing the year of each draw.
    days (np.array): Array representing the day (or period) of each draw.
    time_decay_factor (float): Factor to apply exponential decay to past data.

    Returns:
    tuple: Unique numbers and their estimated probabilities.
    """
    unique, counts = np.unique(numbers, return_counts=True)
    
    if years is not None and days is not None:
        combined_labels = [(y, d) for y, d in zip(years, days)]
        group_unique, group_counts = zip(*Counter(combined_labels).items())
        
        decay_weights = np.array([time_decay_factor ** i for i in range(len(group_counts))])[::-1]
        group_counts = np.array(group_counts) * decay_weights
        
        group_prior = np.ones(len(group_unique)) * alpha
        group_posterior_params = group_prior + group_counts
        group_prob = dirichlet.mean(group_posterior_params)
        alpha = group_prob.mean()

    prior = np.ones_like(counts) * alpha
    posterior_params = prior + counts
    estimated_probabilities = dirichlet.mean(posterior_params)
    
    return unique, estimated_probabilities

def generate_lists(n, main_unique_numbers, main_estimated_probabilities, bonus_unique_numbers, bonus_estimated_probabilities):
    """
    Generates new sets of lottery numbers based on the estimated probabilities.

    Args:
    n (int): Number of sets of lottery numbers to generate.
    main_unique_numbers (np.array): Unique main lottery numbers.
    main_estimated_probabilities (np.array): Estimated probabilities of main lottery numbers.
    bonus_unique_numbers (np.array): Unique bonus lottery numbers.
    bonus_estimated_probabilities (np.array): Estimated probabilities of bonus lottery numbers.

    Returns:
    np.array: A 2D array where each row is a generated set of lottery numbers.
    """
    generated_lists = []
    for _ in range(n):
        main_draw = np.random.choice(main_unique_numbers, size=5, replace=False, p=main_estimated_probabilities)
        main_draw.sort()
        bonus_draw = np.random.choice(bonus_unique_numbers, size=2, replace=False, p=bonus_estimated_probabilities)
        bonus_draw.sort()
        complete_draw = np.concatenate((main_draw, bonus_draw))
        generated_lists.append(complete_draw)
    return np.array(generated_lists)

if __name__ == '__main__':
    numbers_of_lists_to_generate = 1726  # Number of sets of lottery numbers to generate
    data = read_data("training.txt")
    
    # Calculate the number of years and days for each entry based on the assumption of two draws per week
    num_entries = len(data)
    num_years = num_entries // (2 * 52) + 1  # Assuming two draws per week for most years
    years = np.array([num_years - 1 - i // (2 * 52) for i in range(num_entries)])
    days = np.array([1 if i >= 6 * 52 else 0 for i in range(num_entries)])

    main_numbers = data[:, :5].flatten()  # Flatten the array to get main numbers
    bonus_numbers = data[:, 5:].flatten()  # Flatten the array to get bonus numbers

    # Estimate probabilities for main and bonus numbers
    main_unique_numbers, main_estimated_probabilities = estimate_hierarchical_probabilities(
        main_numbers, years=years, days=days, time_decay_factor=0.95)
    bonus_unique_numbers, bonus_estimated_probabilities = estimate_hierarchical_probabilities(
        bonus_numbers, years=years, days=days, time_decay_factor=0.95)

    # Generate new sets of lottery numbers based on the estimated probabilities
    example_generated_lists = generate_lists(
        numbers_of_lists_to_generate, 
        main_unique_numbers, main_estimated_probabilities, 
        bonus_unique_numbers, bonus_estimated_probabilities
    )

    # Format the generated numbers for output to a text file
    formatted_data = ""
    for inner_list in example_generated_lists:
        formatted_line = "".join(format(num, '02d') for num in inner_list)
        formatted_data += formatted_line + '\n'

    # Write the formatted generated numbers to a file
    output_file_path = 'generated_data.txt'
    with open(output_file_path, 'w') as file:
        file.write(formatted_data)
