#%%
import sqlite3
import json
import numpy as np
from collections import defaultdict
from itertools import product
import pickle

#%%
def load_dataframe(fname):
    try:
        return pickle.load(open(fname, 'rb'))
    except:
        raise ValueError('NO DATAFILE FOUND')
    
def get_transition_matrix_entry(memory_string, options, db_fname):
    """Get transition probabilities from database and convert to log probabilities."""
    conn = sqlite3.connect(db_fname)
    cursor = conn.cursor()

    # Check if the table exists
    cursor.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name='transition_matrix';
    """)
    if not cursor.fetchone():
        print("The transition_matrix table does not exist.")
        conn.close()
        return defaultdict(dict)

    # Query only the relevant memory_string
    cursor.execute("""
        SELECT options_string, probability_dict FROM transition_matrix
        WHERE memory_string = ?
    """, (memory_string,))
    rows = cursor.fetchall()
    conn.close()

    # Reconstruct the nested dictionary
    nested_dict = defaultdict(dict)
    for options_string, probability_dict in rows:
        options_dict = json.loads(probability_dict)
        nested_dict[options_string] = {option: options_dict[option] for option in options_dict.keys()}
    
    # Calculate probabilities and convert to log space
    transition_probability = np.array([sum(np.exp([nested_dict[options_string][o] for options_string in nested_dict.keys()])) for o in options])
    transition_probability = transition_probability/sum(transition_probability)
    
    if isinstance(transition_probability[0], float) == False:
        raise ValueError("Transition probability is not a float.")
    
    # Convert to log probabilities
    log_transition_dict = {o: np.log(p) if p > 0 else -np.inf for o, p in zip(options, transition_probability)}
    return log_transition_dict

def create_probability_dict(population, options, db_fname):
    """
    Create a dictionary mapping each microstate to log action probabilities.
    
    Args:
        population (dict): Dictionary of all possible states
        options (list): Possible actions
        db_fname (str): Database filename
        
    Returns:
        dict: Dictionary mapping states to log action probabilities
    """
    log_probability_dict = {}
    for state in population.keys():
        log_probability_dict[state] = get_transition_matrix_entry(memory_string=state, options=options, db_fname=db_fname)
    return log_probability_dict


def instantiate_population(options, memory_size=5):
    """
    Create a dictionary with all possible states as keys, initialized with -inf log probability.
    
    Args:
        options (list): Possible actions
        memory_size (int): Memory size
        
    Returns:
        dict: Dictionary mapping state strings to log proportions (initially -inf)
    """

    all_vectors = []
    for rounds in range(memory_size + 1):
        player1_choices = product(options, repeat=rounds)
        player2_choices = product(options, repeat=rounds)
        for p1, p2 in product(player1_choices, player2_choices):
            all_vectors.append((list(p1), list(p2)))

    # Initialize with -inf in log space (equivalent to 0 in linear space)
    population = {'_'.join(p[0])+'_&_'+'_'.join(p[1]): -np.inf for p in all_vectors}
    return population

def convert_dictionary(original_dict, input_options):
    """
    Convert dictionary from format:
    - Keys: "a1_a2_a3_...&_b1_b2_b3_..." 
    - Values: [logprob1, logprob2]
    
    To format:
    - Keys: "a1_b1_a2_b2_a3_b3_..." with options as 0/1
    - Values: np.exp(logprob1)
    """
    converted_dict = {}
    option_to_number = {option: str(i) for i, option in enumerate(input_options)}
    
    for original_key, original_value in original_dict.items():
        if original_key=='_&_':
            print(original_key, original_value)
            new_key = ''
            new_value = np.exp(original_value[input_options[0]])
            converted_dict[new_key] = new_value
            continue
        # Split the key into player A and player B parts
        player_a_part, player_b_part = original_key.split('&')
    
        a_options = player_a_part.split('_')
        b_options = player_b_part.split('_')
        #print(a_options, b_options)
        a_options.remove('')
        b_options.remove('')
        # Convert options to 0/1
        a_converted = [option_to_number[option] for option in a_options]
        b_converted = [option_to_number[option] for option in b_options]
        
        # Interleave the sequences: a1_b1_a2_b2_...
        max_length = max(len(a_converted), len(b_converted))
        interleaved = []
        
        for i in range(max_length):
            if i < len(a_converted):
                interleaved.append(a_converted[i])
            if i < len(b_converted):
                interleaved.append(b_converted[i])
        
        # Create new key
        new_key = ''.join(interleaved)
        
        # Transform value: np.exp(first element of the list)
        new_value =np.exp(original_value[input_options[0]])
        
        converted_dict[new_key] = new_value
    
    return converted_dict
#%%
options_set_fname = f"crows_pairs_stereo_sample.pkl"
options_dict = load_dataframe(fname = options_set_fname)
options_id_list = list(options_dict.keys())[:11]
#%%
if __name__ == "__main__":
    shorthand = "phi_4"
    memory_size = 5
    for options_id in options_id_list:
        options = options_dict[options_id]['differences']
        matrix_fname = f"matrices/TRANSITION_MATRIX_{shorthand}_ID_{options_id}_-50_100_0.5tmp_BLANK_False.db"

        print(matrix_fname)
        print(options)
        print("\nOption mapping:")
        for i, option in enumerate(options):
            print(f"  '{option}' -> {i}")
        # Initialize population in log space (this is the microstate occupancy vector)
        population = instantiate_population(options, memory_size)

        # Create log probability dictionary    
        log_probability_dict = create_probability_dict(population, options, matrix_fname)
                
        # Convert the dictionary
        converted_dict = convert_dictionary(log_probability_dict, options)

        # save converted_dict to a pickle file
        output_fname = f"q_dicts/Q_dict_{shorthand}_ID_{options_id}_-50_100_0.5tmp_BLANK_False.pkl"
        with open(output_fname, 'wb') as f:
            pickle.dump(converted_dict, f)
