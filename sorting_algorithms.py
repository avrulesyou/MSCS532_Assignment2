import time
import random
import sys

# Set a higher recursion limit for Quick Sort on large inputs
# Default limit (e.g., 1000 or 3000) might be too low for N > 20k
sys.setrecursionlimit(2000000)

# --- Merge Sort Implementation ---
def merge_sort(arr):
    """
    Sorts a list in ascending order using the Merge Sort algorithm.
    Modifies the input list 'arr' in-place.
    """
    if len(arr) > 1:
        mid = len(arr) // 2
        L = arr[:mid]  # Left half
        R = arr[mid:]  # Right half

        merge_sort(L)  # Recursively sort the left half
        merge_sort(R)  # Recursively sort the right half

        # Merge the sorted halves back into arr
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] <= R[j]: # Use <= to maintain stability
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Copy remaining elements of L, if any
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        # Copy remaining elements of R, if any
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
    return arr # Though modified in-place, returning allows for sorted_arr = merge_sort(arr.copy())

# --- Quick Sort Implementation ---
def _median_of_three(arr, low, high):
    """
    Selects a pivot using median-of-three and places it at arr[high].
    This helps avoid worst-case O(n^2) on sorted/nearly-sorted data.
    Returns the pivot value.
    """
    mid = (low + high) // 2
    # Sort arr[low], arr[mid], arr[high] to find median
    if arr[low] > arr[mid]:
        arr[low], arr[mid] = arr[mid], arr[low]
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
    if arr[mid] > arr[high]:
        arr[mid], arr[high] = arr[high], arr[mid]
    # Median is now in arr[mid]. Swap it with arr[high] to use as pivot.
    arr[mid], arr[high] = arr[high], arr[mid]
    return arr[high]

def _partition_lomuto(arr, low, high):
    """
    Partitions the array segment arr[low...high] using Lomuto's scheme.
    Pivot is chosen by _median_of_three and is at arr[high].
    Returns the final index of the pivot.
    """
    pivot_value = _median_of_three(arr, low, high)
    i = low - 1  # Index of smaller element

    for j in range(low, high): # Iterate up to high-1 (arr[high] is pivot)
        if arr[j] <= pivot_value:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1] # Place pivot correctly
    return i + 1

def quick_sort_recursive(arr, low, high):
    """Recursive helper for Quick Sort."""
    if low < high:
        # pi is partitioning index, arr[pi] is now at its sorted place
        pi = _partition_lomuto(arr, low, high)
        quick_sort_recursive(arr, low, pi - 1)
        quick_sort_recursive(arr, pi + 1, high)

def quick_sort(arr):
    """
    Sorts a list in ascending order using the Quick Sort algorithm.
    Uses median-of-three pivot selection and Lomuto's partition scheme.
    Returns a new sorted list (does not modify original 'arr').
    """
    arr_copy = list(arr) # Work on a copy
    quick_sort_recursive(arr_copy, 0, len(arr_copy) - 1)
    return arr_copy

# --- Dataset Generation ---
def generate_random_data(size):
    """Generates a list of 'size' random integers."""
    return [random.randint(0, size * 10) for _ in range(size)]

def generate_sorted_data(size):
    """Generates a sorted list of 'size' integers."""
    return list(range(size))

def generate_reverse_sorted_data(size):
    """Generates a reverse-sorted list of 'size' integers."""
    return list(range(size, -1, -1))

def generate_data_with_duplicates(size, unique_percentage=0.1):
    """Generates a list with a high percentage of duplicate values."""
    if not (0 < unique_percentage <= 1):
        unique_percentage = 0.1 # Default to 10% unique
    num_unique = max(1, int(size * unique_percentage))
    unique_elements = [random.randint(0, size // 2) for _ in range(num_unique)]
    return [random.choice(unique_elements) for _ in range(size)]

# --- Performance Measurement ---
def measure_performance(algorithm_func, data_original, alg_name, data_name, data_size):
    """Measures and prints execution time of a sorting algorithm."""
    print(f"Running {alg_name} on {data_name} data (size {data_size})...")
    data_to_sort = list(data_original) # Use a fresh copy for each sort
    
    start_time = time.perf_counter()
    algorithm_func(data_to_sort) # Execute sort
    end_time = time.perf_counter()
    
    execution_time = end_time - start_time
    print(f"Finished in {execution_time:.6f} seconds.\n")
    
    return {
        "Algorithm": alg_name,
        "Dataset Type": data_name,
        "Size (n)": data_size,
        "Execution Time (s)": execution_time
    }

# --- Main Execution for Comparison ---
if __name__ == "__main__":
    dataset_sizes = [1000, 5000, 10000, 20000, 50000]
    all_performance_results = []

    for size in dataset_sizes:
        print(f"\n--- Generating and testing with dataset size: {size} ---")
        datasets_for_current_size = {
            "Random": generate_random_data(size),
            "Sorted": generate_sorted_data(size),
            "Reverse Sorted": generate_reverse_sorted_data(size),
            "Duplicates (10% unique)": generate_data_with_duplicates(size, 0.1)
        }

        for data_name, current_data_array in datasets_for_current_size.items():
            # Test Merge Sort
            result_ms = measure_performance(merge_sort, current_data_array, "Merge Sort", data_name, size)
            all_performance_results.append(result_ms)
            
            result_qs = measure_performance(quick_sort, current_data_array, "Quick Sort", data_name, size)
            all_performance_results.append(result_qs)

    print("\n--- Aggregate Performance Results Summary ---")
    if all_performance_results:
        headers = all_performance_results[0].keys()
        # Adjust column widths for better readability
        header_fmt_string = "{:<15} | {:<25} | {:<10} | {:<20}"
        row_fmt_string =    "{:<15} | {:<25} | {:<10} | {:<20.6f}"
        
        print(header_fmt_string.format(*headers))
        print("-" * (15 + 25 + 10 + 20 + 9)) # Sum of widths + separators
        
        for res_dict in all_performance_results:
            print(row_fmt_string.format(*(res_dict[h] for h in headers)))

    print("\nNote: Python's built-in list.sort() or sorted() function uses Timsort,")
    print("a highly optimized hybrid algorithm. These implementations are for educational")
    print("purposes to understand Merge Sort and Quick Sort fundamentals.")

    # Code for generating graphs using pandas and matplotlib
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        df = pd.DataFrame(all_performance_results)
        print("\nPandas DataFrame of results:")
        print(df)
    
        pivot_df = df.pivot_table(index='Size (n)', 
                                  columns=['Algorithm', 'Dataset Type'], 
                                  values='Execution Time (s)')
        
        pivot_df.plot(kind='line', marker='o', figsize=(18, 10))
        plt.title('Sorting Algorithm Performance Comparison (Execution Time)')
        plt.ylabel('Execution Time (s) - Logarithmic Scale')
        plt.xlabel('Dataset Size (n)')
        plt.yscale('log') 
        plt.grid(True, which="both", ls="-")
        plt.legend(title='Algorithm & Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        # plt.savefig("performance_comparison_graph.png") # Uncomment to save the graph
        plt.show()
    except ImportError:
        print("\nError: Pandas and/or Matplotlib not installed. Graph generation skipped.")
        print("To generate graphs, please install these packages by running:")
        print("pip install pandas matplotlib")
    except Exception as e:
        print(f"\nAn error occurred during graph generation: {e}")
