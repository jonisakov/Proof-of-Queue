import random
import concurrent.futures
import os



def fill_full_array(n):
    """
    Fills an array with numbers from 1 to ensure the binary tree is full.
    """
    full_size = 1
    while full_size < n:
        full_size = full_size * 2 + 1
    return list(range(1, full_size + 1))

def replace_head_with_children(array):
    """
    Pops the head (root) of the binary tree and recursively replaces it with
    one of its children until reaching the bottom of the tree.
    """
    if not array:
        return None

    i = 0  # Start with the root of the tree
    while True:
        left_child_index = 2 * i + 1
        right_child_index = 2 * i + 2
        if array[i] == None:
            break
        # If the node has no children, stop
        if (left_child_index >= len(array) and right_child_index >= len(array)) or (array[right_child_index] == None and array[left_child_index] == None) :
            array[i] = None
            break

        # Randomly choose the left or right child
        if (right_child_index >= len(array) or array[right_child_index] == None):
            chosen_child_index = left_child_index
        elif(left_child_index >= len(array) or array[left_child_index] == None):
            chosen_child_index = right_child_index
        elif random.choice([True, False]):
            chosen_child_index = left_child_index
        else:
          chosen_child_index = right_child_index

        # Replace the current node with the chosen child
        if chosen_child_index < len(array):
            array[i] = array[chosen_child_index]
            i = chosen_child_index  # Move to the chosen child
        else:
            break

def print_binary_tree_as_array(array):
    """
    Prints the array representation of the binary tree in level order.
    """
    return array

def choose_random_number(max_value):
    """
    Chooses a random number between 1 and the provided max_value.
    
    Args:
        max_value (int): The upper limit for the random number range (inclusive).
    
    Returns:
        int: A random number between 1 and max_value.
    """
    if max_value < 1:
        raise ValueError("The maximum value must be at least 1.")
    return random.randint(1, max_value)


def run_test():
    # Parameters for the run
    array_length = 65536
    rand_value_in_stuck = choose_random_number(array_length)
    array = fill_full_array(array_length)

    counter = 1
    local_run = []

    # Replace the head repeatedly and collect results
    while array[0] is not None:
        if array[0] == rand_value_in_stuck:
            local_run.append([rand_value_in_stuck, counter])
        replace_head_with_children(array)
        counter += 1
    
    return local_run

def main():
    all_runs = []
    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Submit 50,000 tasks
        futures = [executor.submit(run_test) for _ in range(50_000)]
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            all_runs.extend(future.result())

    print("All runs completed.")
    return all_runs

if _name_ == "_main_":
    all_runs = main()