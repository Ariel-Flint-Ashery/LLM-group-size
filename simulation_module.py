#%% imports
import random
import prompting as pr
import pickle
import yaml
from munch import munchify
import utils as ut
import meta_prompting as mp
import sqlite3
import json
from itertools import permutations
from tqdm import tqdm
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%% load running functions
if config.sim.mode == 'api':
    import run_API as ask
if config.sim.mode == 'gpu':
    import run_local as ask

#%% meta prompting
def simulate_meta_prompting(memory_size, rewards, options, fname):
    question_list = ['min', 'max', 'actions', 'payoff', 'round', 'action_i', 'points_i', 'no_actions', 'no_points']
    try:
        tracker = pickle.load(open(fname, 'rb'))
    except:
        tracker = {q: {'responses': [], 'outcome': []} for q in question_list}
    # choose random player
    new_options = options.copy()
    # load their current history up to given round.
    while len(tracker[question_list[0]]['outcome'])<100:
        t = len(tracker[question_list[0]]['outcome'])
        random.shuffle(new_options)
        rules = pr.get_rules(rewards, options = new_options)

        running_player = mp.running_player(options = new_options, memory_size=memory_size, rewards=rewards)
        # get questions
        i, questions, q_list, prompts = mp.get_meta_prompt_list(some_player = running_player, rules=rules, options=new_options)

        # get answers
        for prompt, question, q in zip(prompts, questions, q_list):
            print(f"QUESTION: {question}", flush = True)
            response = ask.get_meta_response(prompt)
            gold_response = mp.gold_sim(q, question, running_player, i, options)
            tracker[q]['responses'].append(response)
            if q == 'actions':
                if all(option in response for option in options):
                    tracker[q]['outcome'].append(1)
                    print('SUCCESS', flush = True)
                else:
                    tracker[q]['outcome'].append(0)
            else:
                print(f"GOLD: {gold_response}", flush = True) 
                if gold_response in response:
                    tracker[q]['outcome'].append(1)
                    print('SUCCESS', flush = True)
                else:
                    tracker[q]['outcome'].append(0)
        print(f"INTERACTION {t}", flush = True)
        f = open(fname, 'wb')
        pickle.dump(tracker, f)
        f.close()
    return tracker

# %%
def sample_from_dict(transition_matrix, options):

    probability_vector = [transition_matrix[option] for option in options]
    return options[ut.log_roulette_wheel(probability_vector)]

# Global connection pool to avoid repeatedly opening/closing connections
_connection_cache = {}

def get_db_connection(fname):
    """
    Get a persistent connection to the database, creating it if necessary.
    """
    global _connection_cache
    
    if fname not in _connection_cache or _connection_cache[fname] is None:
        # Create a new connection
        conn = sqlite3.connect(fname, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")  # Enable concurrent reads while writing
        conn.execute("PRAGMA synchronous=NORMAL;")  # Reduce disk I/O
        conn.execute("PRAGMA cache_size=10000;")   # Increase cache size
        
        # Create table if it doesn't exist
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transition_matrix (
                memory_string TEXT,
                options_string TEXT,
                probability_dict TEXT,
                PRIMARY KEY (memory_string, options_string)
            )
        """)
        conn.commit()
        
        # Store in cache
        _connection_cache[fname] = conn
    
    return _connection_cache[fname]

def close_all_connections():
    """
    Close all database connections.
    Should be called at the end of the program.
    """
    global _connection_cache
    
    for fname, conn in _connection_cache.items():
        if conn is not None:
            try:
                conn.close()
            except Exception as e:
                print(f"Error closing connection to {fname}: {e}")
    
    _connection_cache = {}


def get_transition_matrix(fname, options, my_history, partner_history, prompt, first_target_id_dict):
    """
    Loads or computes a transition probability dictionary using a persistent connection.
    """
    # Create memory key for caching/lookup (preserving original data structure)
    memory_string = '_'.join(my_history) + '_&_' + '_'.join(partner_history)
    options_string = '_'.join(options)

    # Get persistent connection
    conn = get_db_connection(fname)
    cursor = conn.cursor()
    
    try:
        # Check if the entry exists
        cursor.execute("SELECT probability_dict FROM transition_matrix WHERE memory_string = ? AND options_string = ?", 
                    (memory_string, options_string))
        row = cursor.fetchone()

        if row:
            # Load existing probability_dict
            probability_dict = json.loads(row[0])
        else:
            # Compute new transition probabilities (preserving original parameters)
            probability_dict = ask.get_probability_dict(options=options, prompt=prompt, first_target_id_dict=first_target_id_dict)

            # Save to database using a transaction for atomicity
            cursor.execute("INSERT OR REPLACE INTO transition_matrix (memory_string, options_string, probability_dict) VALUES (?, ?, ?)", 
                        (memory_string, options_string, json.dumps(probability_dict)))
            conn.commit()
    except sqlite3.Error as e:
        # Handle potential database errors
        print(f"Database error: {e}")
        conn.rollback()
        raise
        
    return probability_dict


def get_full_transition_matrix(options, memory_size, rewards, options_id):
    first_target_id_dict = ask.encode_decode_options(options = options)
    # get empty memory transitions
    action_vectors = ut.generate_action_vectors(options = options, memory_size = memory_size)
    all_options = list(permutations(options))
    for opts in all_options:
        opts = list(opts)
        for pair in tqdm(action_vectors):
            player=ut.get_player()
            m = len(pair[0])
            for h in range(m):
                my_answer, partner_answer = [p[h] for p in pair]
                ut.update_dict(player, my_answer, partner_answer, ut.get_outcome(my_answer,partner_answer, rewards))

            rules = pr.get_rules(rewards, options = opts)
            # get prompt with rules & history of play
            prompt = pr.get_prompt(player = player, memory_size=m, rules = rules)
            # get agent response
            matrix_fname = f"matrices/TRANSITION_MATRIX_{config.model.shorthand}_ID_{options_id}_{'_'.join([str(r) for r in rewards])}_{config.params.temperature}tmp_BLANK_{config.sim.fill_blank}.db"
            matrix = get_transition_matrix(fname = matrix_fname, options = opts, my_history=player['my_history'], partner_history=player['partner_history'], prompt = prompt, first_target_id_dict=first_target_id_dict)