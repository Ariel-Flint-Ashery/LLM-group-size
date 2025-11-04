import numpy as np
import random
from numba import float64, int64
from numba.experimental import jitclass
import pickle
from tqdm import tqdm, trange
from copy import deepcopy
import multiprocessing as mp
import os
import shutil

#%% BASICS
def shift(key, choice, nn_choice, H):
    """
        key grows until it reaches the length 2*H, then it shifts to the left

    """

    if len(key) < 2*H:
        return key + str(choice) + str(nn_choice)
    else:
        return key[2:] + str(choice) + str(nn_choice)

# Random mapping beteween states and integers

def integer_mapping(keys_H):
    """
        Create a random mapping between the states and integers
        Returns:
            F: dictionary mapping states to integers
            Finv: dictionary inverse mapping integers to states

    """        

    F = {} # direct function
    Finv = {} # inverse function

    item = 0

    for k in keys_H:
        F[k] = item
        Finv[item] = k
        item += 1

    return F, Finv

def state_transitions(q_H, H, F, Finv):
    """
        Compute the transition tensor P
        Returns:
            P_key: state of the transition
            P_value: probability of the transition
    """

    n = len(q_H)

    P_value = np.zeros((n, n, 4), dtype=float) # Store as a numpy array
    P_key = np.zeros((n, n, 4), dtype=int) # Store as a numpy array

    for i in range(n):
        for j in range(n):
            key_i = Finv[i]
            key_j = Finv[j]

            prob_0_i = q_H[key_i]
            prob_0_j = q_H[key_j]

            next_i_1 = F[shift(key_i, 0, 0, H)]
            next_i_2 = F[shift(key_i, 0, 1, H)]
            next_i_3 = F[shift(key_i, 1, 0, H)]
            next_i_4 = F[shift(key_i, 1, 1, H)]


            P_key[i,j,0] = next_i_1
            P_key[i,j,1] = next_i_2
            P_key[i,j,2] = next_i_3
            P_key[i,j,3] = next_i_4
            P_value[i,j,0] = prob_0_i*prob_0_j
            P_value[i,j,1] = prob_0_i*(1-prob_0_j)
            P_value[i,j,2] = (1-prob_0_i)*prob_0_j
            P_value[i,j,3] = (1-prob_0_i)*(1-prob_0_j)  

    return P_key, P_value

def integer_probabilities(q_H, F):
    """
        Compute the probability of output 0 for each state
        Returns:
            q_s: probability of output 0 for each state
    """
    n = len(q_H)

    q_s = np.zeros(n)
    for k in F:
        i = F[k]

        q_s[i] = q_H[k]

    return q_s

def H_reducer(q_total, H):
    """
        Reduce the dictionary to the keys of length 2*H
    """
    q_H = {k: v for k, v in q_total.items() if len(k) <= 2*H}

    keys_H = set(q_H.keys())

    steady_0 = ''.join(['0']*(2*H))
    steady_1 = ''.join(['1']*(2*H))

    return q_H, keys_H, steady_0, steady_1

spec = [
    ('N', int64),
    ('q_s', float64[:]),
    ('population', int64[:]),
    ('P_key', int64[:,:,:]),
    ('empty_state', int64)
]

@jitclass(spec)
class LLM_dynamics_integer:
    """
        Simple class simulator
    """
    def __init__(self, N, q_s, P_key, empty_state):
        self.N = N
        self.q_s = q_s
        self.population = np.zeros(N, dtype=np.int64)
        self.P_key = P_key
        self.empty_state = empty_state
        
    def initialize_population_zero(self):
        """
            Initialize the population of LLMs to the empty state
        """
        
        for i in range(self.N):
            self.population[i] = self.empty_state

    def initialize_population_random(self):
        """
            Initialize the population of LLMs
        """

        n  = len(self.q_s)
        
        for i in range(self.N):
            k = random.randint(0, n-1)
            self.population[i] = k

    def update(self):
        """
            Update the population of LLMs one Monte Carlo time step
        """
        success = 0
        all_words = 0
        for s in range(self.N):
            # choose two different LLM
            i = random.randint(0, self.N-1)
            j = random.randint(0, self.N-1)
            while j == i:
                j = random.randint(0, self.N-1)

            prob_i = self.q_s[self.population[i]]
            prob_j = self.q_s[self.population[j]]

            rr = random.random()
            first = 1 - int( rr < prob_i)

            rr = random.random()
            second = 1 - int( rr < prob_j)

            k_i = 2*first + second
            k_j = 2*second + first

            temp_i = self.population[i]
            temp_j = self.population[j]
            self.population[i] = self.P_key[temp_i, temp_j, k_i]
            self.population[j] = self.P_key[temp_j, temp_i, k_j]
            all_words += first + second
            success += first == second

        return success/self.N, all_words/(2*self.N)

def run_simulation(fname, data_dict, q_total, options, Ns = None, H = 5, ITERS=1000, TIME=1000):
    H = 5

    q_H, keys_H, steady_0, steady_1 = H_reducer(q_total, H)

    F, Finv = integer_mapping(keys_H)

    q_s = integer_probabilities(q_H, F)

    empty_state = F['']

    P_key, P_value = state_transitions(q_H, H, F, Finv)

    if Ns is None:
        Ns = np.geomspace(10, 10000, 25, dtype=int)

    for N in Ns:
        if N in data_dict.keys():
            ITERS_TO_RUN = ITERS - data_dict[N]['total_attempts']
            if ITERS_TO_RUN <= 0:
                print("Skipping N = {}".format(N))
                continue
            else:
                print("Resuming N = {}; Already ran {} iterations".format(N, ITERS - ITERS_TO_RUN))
        else:
            print("Creating new entry for N = {}".format(N))
            ITERS_TO_RUN = ITERS
            data_dict[N] = {0: {'consensus': 0, 'time': [], 'success': [], 'words': []},
                       1: {'consensus': 0, 'time': [], 'success': [], 'words': []}, 
                       'non_consensus': {0: {'consensus': 0, 'time': [], 'success': [], 'words': []},
                       1: {'consensus': 0, 'time': [], 'success': [], 'words': []}},
                       'total_attempts': 0}
        
        data = data_dict[N]

        for it in trange(ITERS_TO_RUN, desc="Process: {}; N: {}; Pair: {}".format(os.getpid(), N, options)):
            llm = LLM_dynamics_integer(N, q_s, P_key, empty_state)
            llm.initialize_population_zero()
            all_word_tracker = []
            all_success_tracker = []
            success_tracker = []
            words_tracker = []
            for t in range(TIME):
                # One Monte Carlo time step = N individual updates
                success, words = llm.update()
                all_word_tracker.append(words)
                all_success_tracker.append(success)
                if len(success_tracker) < 3:
                    success_tracker.append(success)
                    words_tracker.append(words)
                else:
                    # pop the first element
                    success_tracker.pop(0)
                    words_tracker.pop(0)
                    success_tracker.append(success)
                    words_tracker.append(words)

                    if np.mean(success_tracker) >= 0.98:
                        final_state = int(np.mean(words_tracker) >= 0.5)
                        data[final_state]['time'].append(t)
                        data[final_state]['consensus'] += 1
                        data[final_state]['success'].append(all_success_tracker)
                        data[final_state]['words'].append(all_word_tracker)
                        break
                if t == TIME - 1:
                    final_state = int(np.mean(words_tracker) >= 0.5)
                    data['non_consensus'][final_state]['time'].append(t)
                    data['non_consensus'][final_state]['consensus'] += 1
                    data['non_consensus'][final_state]['success'].append(all_success_tracker)
                    data['non_consensus'][final_state]['words'].append(all_word_tracker)


        num_0 = data[0]['consensus']
        num_1 = data[1]['consensus']

        if num_0 + num_1 != ITERS:
            print("Not all the simulations converged")
        data['total_attempts'] += ITERS_TO_RUN
        data_dict[N] = data
        # Data validation check
        total_simulations = data[0]['consensus'] + data[1]['consensus']
        total_times = len(data[0]['time']) + len(data[1]['time'])
        if total_times != total_simulations:
            print(f"WARNING: Data inconsistency for N={N}! "
                f"Total consensus: {total_simulations}, Total times: {total_times}")
            
        # save data_dict with process-safe filename
        temp_fname = f"{fname}.tmp_{os.getpid()}"
        with open(temp_fname, 'wb') as f:
            pickle.dump(data_dict, f)

        # Atomic move to final filename (more process-safe)
        shutil.move(temp_fname, fname)


#%% LOAD q_total FROM FILES
def load_dataframe(fname):
    try:
        return pickle.load(open(fname, 'rb'))
    except:
        raise ValueError('NO DATAFILE FOUND')

def run_single_option(args):
    """
    Function to run simulation for a single options_id.
    This function will be called by each process.
    """
    options_id, options_dict, shorthand, H, Ns = args
    process_seed = os.getpid() + options_id  # Unique seed per process
    random.seed(process_seed)
    np.random.seed(process_seed)
    try:
        options = options_dict[options_id]['differences']
        q_dict_fname = f"q_dicts/Q_dict_{shorthand}_ID_{options_id}_-50_100_0.5tmp_BLANK_False.pkl"
        
        print(f"Process {os.getpid()}: Processing options_id {options_id}")
        print(f"Options: {options}")
        print("\nOption mapping:")
        for i, option in enumerate(options):
            print(f"  '{option}' -> {i}")
        
        with open(q_dict_fname, 'rb') as f:
            q_total = pickle.load(f)
        
        output_file = f"LLM_dynamics_{shorthand}_ID_{options_id}_-50_100_H_{H}_0.5tmp.pkl"
        
        try:
            with open(output_file, 'rb') as f:
                data_dict = pickle.load(f)
        except:
            data_dict = {}

        print("Previously saved data points:")
        print(data_dict.keys())

        run_simulation(output_file, data_dict, q_total, options=options, Ns=Ns, H=H, ITERS=1000, TIME=1000)

        return f"Successfully completed options_id {options_id}"
        
    except Exception as e:
        return f"Error processing options_id {options_id}: {str(e)}"

def run_parallel_simulations(max_processes=None):
    """
    Run simulations in parallel for different options_id values.
    
    Parameters:
    max_processes: Maximum number of processes to use. If None, uses all available CPU cores.
    """
    options_set_fname = f"crows_pairs_stereo_sample.pkl"
    options_dict = load_dataframe(fname=options_set_fname)
    indices = [i for i in range(11)]# if i not in [10]]
    options_id_list = [list(options_dict.keys())[i] for i in indices]
    shorthand = "QWQ_32B"
    H = 5
    Ns = [2,4,6,8,10,20,40,60,80,100,200,300,400, 500] 

    # Determine number of processes to use
    if max_processes is None:
        max_processes = mp.cpu_count()-1  # Leave one core free
    
    # Limit processes to available options or CPU cores, whichever is smaller
    num_processes = min(max_processes, len(options_id_list))
    
    print(f"Running simulations on {num_processes} processes for {len(options_id_list)} options")
    print(f"Available CPU cores: {mp.cpu_count()}")
    
    # Prepare arguments for each process
    process_args = [(options_id, options_dict, shorthand, H, Ns) for options_id in options_id_list]
    
    # Create and start processes
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_single_option, process_args)
    
    # Print results
    print("\n" + "="*50)
    print("SIMULATION RESULTS:")
    print("="*50)
    for result in results:
        print(result)

if __name__ == "__main__":
    # You can specify the maximum number of processes to use
    # If you want to use all available CPU cores, use None
    # If you want to limit the number of processes, specify a number
    run_parallel_simulations(max_processes=None)
    