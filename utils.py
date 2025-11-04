import random
import pickle
import yaml
from munch import munchify
from scipy.special import logsumexp
import numpy as np
from itertools import product
import string
#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
#%%
def load_mainframe(fname):
    try:
        mainframe = pickle.load(open(fname, 'rb'))
    except:
        mainframe = dict()
    
    return mainframe

def get_player():
    return {'my_history': [], 'partner_history': [], 'interactions': [], 'score': 0, 'score_history': [], 'outcome': []}

def get_outcome(my_answer, partner_answer, rewards):
    if my_answer == partner_answer:
        return rewards[1]
    return rewards[0]

def update_dict(player, my_answer, partner_answer, outcome):
  player['score'] += outcome
  player['my_history'].append(my_answer)
  player['partner_history'].append(partner_answer)
  player['score_history'].append(player['score'])
  player['outcome'].append(outcome)

  return player

def get_random_prepared_player(history, rewards):
    #print("CREATING RANDOM PREPARED PLAYER")
    dataframe = get_player()
    for h in history:
        my_answer, partner_answer = h       
        update_dict(dataframe, my_answer, partner_answer, get_outcome(my_answer,partner_answer, rewards))
    # print("---------- CREATING NEW INITIALISED DATAFRAME ----------")
    # print(dataframe['simulation'])
    return dataframe

def has_tracker_converged(tracker, N, threshold = config.params.convergence_threshold):
    if len(tracker['answers']) < 3*N:
      return False
    else:
      history = [x for xs in tracker['answers'][-3*N:] for x in xs]
      word = max(set(history), key = history.count)
      if history.count(word)/len(history) < threshold:
        return False
      return True
    # if sum(tracker['outcome'][-config.params.convergence_time:]) < threshold*config.params.convergence_time:
    #     return False
    # return True

def update_tracker(tracker, p1, p2, p1_answer, p2_answer, outcome):
  tracker['players'].append([p1, p2])
  tracker['answers'].append([p1_answer, p2_answer])
  if outcome > 5:
    tracker['outcome'].append(1)
  else:
    tracker['outcome'].append(0)

def normalize_logprobs(logprobs):
    logtotal = logsumexp(logprobs) #calculates the summed log probabilities
    normedlogs = []
    for logp in logprobs:
        normedlogs.append(logp - logtotal) #normalise - subtracting in the log domain equivalent to divising in the normal domain
    return normedlogs

def normalize_probs(probs):
    total = sum(probs) #calculates the summed probabilities
    normedprobs = []
    for p in probs:
        normedprobs.append(p / total) 
    return normedprobs

def roulette_wheel(normedprobs):
    r=random.random() #generate a random number between 0 and 1
    accumulator = normedprobs[0]
    for i in range(len(normedprobs)):
        if r < accumulator:
            return i
        accumulator = accumulator + normedprobs[i + 1]

def log_roulette_wheel(logprobs):
    return np.argmax(np.array(logprobs) + np.random.gumbel(size = len(logprobs)))

def generate_action_vectors(memory_size=5, options=[0, 1]):
    all_vectors = []
    for rounds in range(memory_size + 1):
        player1_choices = product(options, repeat=rounds)
        player2_choices = product(options, repeat=rounds)
        for p1, p2 in product(player1_choices, player2_choices):
            all_vectors.append((list(p1), list(p2)))
    return all_vectors

def generate_unique_strings(n, k=3):
    if k < 3:
        raise ValueError("k must be at least 3 to satisfy the constraints.")
    
    unique_strings = set()
    characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    
    while len(unique_strings) < n:
        # Ensure we include at least one number, one lowercase, and one uppercase letter
        num = random.choice(string.digits)
        lower = random.choice(string.ascii_lowercase)
        upper = random.choice(string.ascii_uppercase)
        
        # Generate the remaining characters while ensuring uniqueness
        remaining_chars = random.sample(
            list(set(characters) - {num, lower, upper}), max(0, k - 3)
        )
        
        # Combine all characters and shuffle
        all_chars = [num, lower, upper] + remaining_chars
        random.shuffle(all_chars)
        
        # Create the string and add it to the set
        random_string = ''.join(all_chars)
        unique_strings.add(random_string)
    
    return list(unique_strings)
