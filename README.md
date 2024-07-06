Lottery Number Generator
This project is designed to read historical lottery data, estimate the probabilities of drawing certain numbers using hierarchical Bayesian methods, and generate new sets of lottery numbers based on these estimated probabilities. The hierarchical probabilities consider time decay, giving more weight to recent data.

Table of Contents
Prerequisites
Installation
Usage
File Descriptions
Functions
License
Prerequisites
Python 3.x
Required Python libraries: numpy, scipy
You can install the necessary libraries using pip:

bash
Copy code
pip install numpy scipy
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/lottery-number-generator.git
cd lottery-number-generator
Ensure you have the training.txt file with historical lottery data in the same directory. Each line in the file should represent a set of lottery numbers without delimiters.

Usage
Run the script to generate lottery numbers:

bash
Copy code
python generate_lottery_numbers.py
The generated lottery numbers will be saved in a file named generated_data.txt.

File Descriptions
generate_lottery_numbers.py: Main script that reads the data, estimates probabilities, and generates new lottery numbers.
training.txt: Input file containing historical lottery data.
Functions
read_data(file_path)
Reads the lottery data from the file and structures it into a 2D numpy array.

Parameters:

file_path (str): Path to the file containing lottery data.
Returns:

np.array: A 2D numpy array where each row represents a set of lottery numbers.
estimate_hierarchical_probabilities(numbers, alpha=1, years=None, days=None, time_decay_factor=0.9)
Estimates the probabilities of drawing certain numbers using a hierarchical Bayesian method.

Parameters:

numbers (np.array): Flattened array of lottery numbers.
alpha (float): Dirichlet prior parameter.
years (np.array): Array representing the year of each draw.
days (np.array): Array representing the day (or period) of each draw.
time_decay_factor (float): Factor to apply exponential decay to past data.
Returns:

tuple: Unique numbers and their estimated probabilities.
generate_lists(n, main_unique_numbers, main_estimated_probabilities, bonus_unique_numbers, bonus_estimated_probabilities)
Generates new sets of lottery numbers based on the estimated probabilities.

Parameters:

n (int): Number of sets of lottery numbers to generate.
main_unique_numbers (np.array): Unique main lottery numbers.
main_estimated_probabilities (np.array): Estimated probabilities of main lottery numbers.
bonus_unique_numbers (np.array): Unique bonus lottery numbers.
bonus_estimated_probabilities (np.array): Estimated probabilities of bonus lottery numbers.
Returns:

np.array: A 2D array where each row is a generated set of lottery numbers.
