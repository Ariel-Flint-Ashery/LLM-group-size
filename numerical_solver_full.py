
#%%
from tqdm import tqdm
from itertools import product
import numpy as np
from scipy.special import logsumexp  # For numerically stable log-space operations
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
import sqlite3 
import json
import pickle

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

def update_initial_population(population, initial_conditions=None):
    """
    Update population with initial conditions in log space.
    
    Args:
        population (dict): Current population dictionary (log probabilities)
        initial_conditions (dict, optional): Dictionary mapping states to initial proportions
        
    Returns:
        dict: Updated population dictionary (log probabilities)
    """
    if initial_conditions is None:
        # Generate random values and normalize in log space
        v = np.random.normal(0, 1, len(population.keys()))
        # Take absolute values to ensure non-negative probabilities
        v = np.abs(v)
        # Convert to linear space, normalize, then back to log space
        v_norm = v / np.sum(v)
        for i, k in enumerate(population.keys()):
            population[k] = np.log(v_norm[i]) if v_norm[i] > 0 else -np.inf
    else:
        # Reset all population values to -inf (log of 0)
        for k in population.keys():
            population[k] = -np.inf
        
        # Add the specified initial conditions in log space
        total = sum(initial_conditions.values())
        if total <= 0:
            raise ValueError("Initial conditions must sum to a positive value")
            
        for k, v in initial_conditions.items():
            if k in population:
                population[k] = np.log(v / total) if v > 0 else -np.inf
    
    # Verify normalization by converting from log space, summing, then checking
    # We use logsumexp for numerical stability
    log_sum = logsumexp(list(population.values()))
    if not np.isclose(np.exp(log_sum), 1.0):
        # Normalize in log space by subtracting log_sum
        for k in population:
            population[k] -= log_sum
            
    return population

def get_possible_transitions(microstate, options, memory):
    """
    Given a microstate string, return all possible successor microstates.
    """
    my_hist_str, partner_hist_str = microstate.split('_&_')
    my_hist = my_hist_str.split('_')
    partner_hist = partner_hist_str.split('_')

    # assert len(my_hist) == memory and len(partner_hist) == memory, "History length mismatch."

    if len(my_hist) == memory:
        # Remove the oldest memory (leftmost), keep the most recent (rightmost)
        my_hist_trimmed = my_hist[1:]
        partner_hist_trimmed = partner_hist[1:]
    elif '' in my_hist:
        my_hist_trimmed = []
        partner_hist_trimmed = []
    else:
        my_hist_trimmed = my_hist
        partner_hist_trimmed = partner_hist

    transitions = []
    for my_action in options:
        for partner_action in options:
            new_my_hist = my_hist_trimmed + [my_action]
            new_partner_hist = partner_hist_trimmed + [partner_action]
            new_microstate = '_'.join(new_my_hist) + '_&_' + '_'.join(new_partner_hist)
            transitions.append(new_microstate)

    return transitions

def get_first_state(options, initial_productions, memory_size=1):
    """
    Create an initial population distribution based on initial production probabilities.
    
    Args:
        options (list): List of possible actions (e.g., ['Black', 'White'])
        initial_productions (list): List of initial production probabilities for each option
        memory_size (int): Size of memory for each player (default: 1)
        
    Returns:
        dict: Initial population distribution in log space
    """
    # Validate inputs
    if len(options) != len(initial_productions):
        raise ValueError("options and initial_productions must have the same length")
    
    # Create a mapping from option to production probability for easier lookup
    production_map = {option: prob for option, prob in zip(options, initial_productions)}
    assert sum(initial_productions) == 1.0
    # Create all possible state combinations
    population = {}
    for p1_choices in product(options, repeat=memory_size):
        for p2_choices in product(options, repeat=memory_size):
            # Create state key
            state_key = '_'.join(p1_choices) + '_&_' + '_'.join(p2_choices)
            
            # Calculate log probability based on initial productions
            # Only use the first action from each player's history if memory_size > 1
            p1_first_action = p1_choices[0]
            p2_first_action = p2_choices[0]
            
            # Use production probabilities to compute joint probability
            joint_prob = production_map[p1_first_action] * production_map[p2_first_action]
            
            # Convert to log space, handling zero probabilities
            if joint_prob > 0:
                population[state_key] = joint_prob#np.log(joint_prob)
            else:
                population[state_key] = 0.0#-np.inf
    
    return population

def calculate_population_action_probability(action, x_t, log_probability_dict):
    """
    Calculate the log probability that a random agent from the population would choose 'action'.
    
    Args:
        action (str): The action we want to calculate probability for (e.g., 'Black')
        x_t (dict): Current population distribution in log space
        log_probability_dict (dict): Log action probability dictionary
        
    Returns:
        float: Log probability of observing this action from the population
    """
    # For each state with non-zero probability, compute log(proportion) + log(prob of action)
    # Then use logsumexp to add these terms in log space
    log_probs = []
    
    for state, log_proportion in x_t.items():
        if log_proportion > -np.inf:  # Equivalent to proportion > 0 in linear space
            log_probs.append(log_proportion + log_probability_dict[state][action])
    
    # If no valid states, return -inf (log of 0)
    if not log_probs:
        return -np.inf
        
    # logsumexp computes log(sum(exp(log_probs))) in a numerically stable way
    return logsumexp(log_probs)

def calculate_transition_probabilities(source_state, target_states, x_t, log_probability_dict):
    """
    Calculate log transition probabilities from source_state to each target state given the
    current population distribution x_t.
    
    Args:
        source_state (str): The current state
        target_states (list): List of possible next states
        x_t (dict): Current population distribution across states (in log space)
        log_probability_dict (dict): Maps states to log action probabilities
        
    Returns:
        dict: Mapping from target states to log transition probabilities
    """
    log_transition_probs = {}
    
    # For each possible target state
    for target_state in target_states:
        # Parse the target state to extract the new actions
        target_my_hist_str, target_partner_hist_str = target_state.split('_&_')
        target_my_hist = target_my_hist_str.split('_')
        target_partner_hist = target_partner_hist_str.split('_')
        
        # Extract the most recent actions (what we're adding in this transition)
        my_new_action = target_my_hist[-1]
        partner_new_action = target_partner_hist[-1]
        
        # Log probability that I choose my_new_action given my current state
        my_log_action_prob = log_probability_dict[source_state][my_new_action]
        
        # Log probability that a random partner from the population chooses partner_new_action
        partner_log_action_prob = calculate_population_action_probability(partner_new_action, x_t, log_probability_dict)
        
        # Combined log probability of this transition (sum in log space = product in linear space)
        log_transition_probs[target_state] = my_log_action_prob + partner_log_action_prob
    
    return log_transition_probs

def evolve_state(x_t, options, memory, log_probability_dict):
    """
    Evolve the state distribution vector x_t to x_{t+1} in log space
    
    Args:
        x_t (dict): Maps state strings to log proportions
        options (list): Possible actions
        memory (int): Memory size
        log_probability_dict (dict): Log action probability dictionary
        
    Returns:
        dict: The new state distribution x_{t+1} in log space
    """
    x_next = {state: -np.inf for state in x_t.keys()}  # Initialize with -inf (log of 0)
    
    # For each current state and its log proportion
    for source_state, source_log_prop in x_t.items():
        if source_log_prop > -np.inf:  # Equivalent to source_prop > 0 in linear space
            # Get possible transitions from this state
            possible_targets = get_possible_transitions(source_state, options, memory)
            
            # Calculate log transition probabilities based on current distribution
            log_transition_probs = calculate_transition_probabilities(source_state, possible_targets, x_t, log_probability_dict)
            
            # Distribute the source population according to transition probabilities
            for target_state, log_prob in log_transition_probs.items():
                # In log space: log(x_next[target] + source_prop * prob)
                # = log(exp(log(x_next[target])) + exp(log(source_prop) + log(prob)))
                if x_next[target_state] == -np.inf:
                    # If x_next[target_state] is 0 (-inf in log space), just assign the new value
                    x_next[target_state] = source_log_prop + log_prob
                else:
                    # Otherwise, need to add the probabilities in log space
                    x_next[target_state] = logsumexp([x_next[target_state], source_log_prop + log_prob])
    
    # Normalize the distribution in log space
    non_inf_values = [v for v in x_next.values() if v > -np.inf]
    if non_inf_values:  # Check if any non-zero probabilities exist
        log_sum = logsumexp(non_inf_values)
        for state in x_next:
            if x_next[state] > -np.inf:
                x_next[state] -= log_sum
    
    return x_next

def create_probability_dict(population, options, db_fname):
    """
    Create a dictionary mapping each state to log action probabilities.
    
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

def run_simulation(options, matrix_fname, memory_size=5, timesteps=100, initial_conditions=None):
    """
    Run a simulation of the state evolution system using log probabilities.
    
    Args:
        options (list): Possible actions
        matrix_fname (str): Database filename
        memory_size (int): Memory size
        timesteps (int): Number of timesteps to simulate
        initial_conditions (dict, optional): Dictionary mapping states to initial proportions
        
    Returns:
        tuple: (population_history, action_probabilities_history)
    """
    # Initialize population in log space
    population = instantiate_population(options, memory_size)
    population = update_initial_population(population, initial_conditions)
    
    # Create log probability dictionary    
    log_probability_dict = create_probability_dict(population, options, matrix_fname)
    print("Log probability dictionary created.")
    
    # Track history (store log probabilities in history)
    population_history = [deepcopy(population)]
    action_probabilities_history = []
    
    # Track the probability of choosing each action at each timestep
    for t in tqdm(range(timesteps)):
        # Calculate current log probability of each action
        log_action_probs = {action: calculate_population_action_probability(action, population, log_probability_dict) 
                            for action in options}
        
        # Convert log probabilities to linear for storage/plotting
        action_probs = {action: np.exp(log_prob) if log_prob > -np.inf else 0.0 
                       for action, log_prob in log_action_probs.items()}
        action_probabilities_history.append(action_probs)
        
        # Evolve the state
        population = evolve_state(population, options, memory_size, log_probability_dict)
        population_history.append(deepcopy(population))
    
    return population_history, action_probabilities_history

def convert_log_to_linear(log_population):
    """
    Convert a population from log space to linear space for visualization.
    
    Args:
        log_population (dict): Population in log space
        
    Returns:
        dict: Population in linear space
    """
    return {state: np.exp(log_prop) if log_prop > -np.inf else 0.0 
            for state, log_prop in log_population.items()}

def plot_action_probabilities(action_probabilities_history, options):
    """
    Plot the probability of each action over time.
    
    Args:
        action_probabilities_history (list): List of dictionaries mapping actions to probabilities
        options (list): Possible actions
    """
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(10, 8))
    colors = ['tab:red', 'tab:blue']
    for i, action in enumerate(options):
        probabilities = [step[action] for step in action_probabilities_history]
        ax.plot(probabilities, label=action, color =colors[i] )

    ax.set_xlabel('Timestep', fontsize = 20)
    ax.set_ylabel('Norm Production Probability', fontsize = 20)
    #ax.set_title('Action Probabilities Over Time', fontsize = 18)
    ax.legend(fontsize = 16)
    ax.set_ylim(top = 1.1, bottom = 0.0)
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(3)
    for axis in ['right','top']:
        ax.spines[axis].set_visible(False)
    #axs.set_xlim(left=0, right = 45)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=18, length=8) 
    ax.tick_params(width=3, length=10)
    #plt.savefig('action_probabilities.pdf', dpi = 300)
    plt.show()

def plot_state_evolution(population_history, top_n=5):
    """
    Plot the evolution of the top N states over time.
    
    Args:
        population_history (list): List of population dictionaries at each timestep (in log space)
        top_n (int): Number of top states to plot
    """
    # Convert final population to linear space for sorting
    final_log_population = population_history[-1]
    final_population = convert_log_to_linear(final_log_population)
    
    # Find the states with highest average proportion
    top_states = sorted(final_population.keys(), key=lambda k: final_population[k], reverse=True)[:top_n]
    
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(12, 8))
    
    for state in top_states:
        # Convert all populations to linear space for plotting
        proportions = [np.exp(pop[state]) if pop[state] > -np.inf else 0.0 for pop in population_history]
        ax.plot(proportions, label=f'{state}')
    
    for axis in ['left','bottom']:
        ax.spines[axis].set_linewidth(3)
    for axis in ['right','top']:
        ax.spines[axis].set_visible(False)
    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax.set_xlabel('Timestep', fontsize = 18)
    ax.set_ylabel('Proportion', fontsize = 18)
    ax.set_title(f'Evolution of Top {top_n} Memory Configurations', fontsize = 18)
    ax.legend(fontsize = 10)
    #plt.grid(True)
    #plt.savefig('state_evolution.pdf', dpi = 300)
    plt.show()
#%%
# Example usage
if __name__ == "__main__":
    memory_size = 5
    timesteps = 1000
    num_repeats = 1 # simulation is deterministic. If initial conditions are random, then set >1
    options_set_fname = f"crows_pairs_stereo_sample.pkl"
    options_dict = load_dataframe(fname = options_set_fname)
    options_id_list = list(options_dict.keys())[:11]
    # Define initial conditions (optional)
    # This starts with 100% in one specific state
    initial_state = '_&_'
    initial_conditions = {initial_state: 1.0}
    shorthand = "QWQ_32B"

    # Run simulation
    population_history_tracker = []
    action_probabilities_history_tracker = []
    for x in range(num_repeats):
        for i in range(11):
            options_id = options_id_list[i]
            options = options_dict[options_id]['differences']
            matrix_fname = f"matrices/TRANSITION_MATRIX_{shorthand}_ID_{options_id}_-50_100_0.5tmp_BLANK_False.db" #f"matrices/TRANSITION_MATRIX_llama31_70B_ID_102_-50_100_0.5tmp_BLANK_False.db"
            print(f"STARTING SIMULATION | MODEL {shorthand} | OPTIONS ID {options_id} | OPTIONS {options} ")
            population_history, action_probabilities_history = run_simulation(
                options,
                matrix_fname=matrix_fname,
                memory_size=memory_size,
                timesteps=timesteps,
                initial_conditions=initial_conditions
            )
            population_history_tracker.append(population_history)
            action_probabilities_history_tracker.append(action_probabilities_history)

            f = open(f"{shorthand}_{options_id}_numerical_simulation_action_probability_history.pkl", 'wb')
            pickle.dump(action_probabilities_history, f)
            f.close()

            # Save log probabilities as well (for future use with log-based calculations)
            f = open(f"{shorthand}_{options_id}_numerical_simulation_log_population_history.pkl", 'wb')
            pickle.dump(population_history, f)
            f.close()

# %%
