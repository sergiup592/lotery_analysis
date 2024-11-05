import numpy as np

class LotteryDataError(Exception):
    """Custom exception for lottery data operations"""
    pass

def validate_numbers(numbers: np.ndarray, n_main: int = 6, n_bonus: int = 1) -> bool:
    """Basic validation of lottery numbers"""
    if len(numbers) != n_main + n_bonus:
        return False
    return True

def load_data(file_path: str, n_main: int = 6, n_bonus: int = 1) -> np.ndarray:
    """Load and validate lottery data"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        structured_data = []
        current_numbers = []
        expected_numbers = n_main + n_bonus
        
        for line in lines:
            try:
                num = int(line.strip())
                current_numbers.append(num)
                if len(current_numbers) == expected_numbers:
                    # Convert to numpy array for validation
                    numbers_array = np.array(current_numbers)
                    if validate_numbers(numbers_array, n_main, n_bonus):
                        structured_data.append(current_numbers)
                    current_numbers = []
            except ValueError:
                continue
                
        if not structured_data:
            raise LotteryDataError("No valid lottery data found")
            
        return np.array(structured_data)
        
    except FileNotFoundError:
        raise LotteryDataError(f"Data file not found: {file_path}")
    except Exception as e:
        raise LotteryDataError(f"Error reading data: {str(e)}")

def save_data(data: np.ndarray, file_path: str) -> None:
    """Save lottery data to a text file with numbers in continuous format"""
    try:
        with open(file_path, 'w') as f:
            for draw in data:
                # Format each number to ensure 2 digits (01, 02, etc.)
                formatted_numbers = ''.join(f"{num:02d}" for num in draw)
                f.write(f"{formatted_numbers}\n")
    except Exception as e:
        raise LotteryDataError(f"Error saving data: {str(e)}")

if __name__ == "__main__":
    try:
        # Load the data
        loaded_data = load_data("lottery_numbers.txt")
        print(f"Loaded {len(loaded_data)} draws")
        print("First draw:", loaded_data[0])
        
        # Save to new file
        save_data(loaded_data, "lottery_numbers_formatted.txt")
        print("Data saved successfully")
        
    except LotteryDataError as e:
        print(f"Error: {e}")