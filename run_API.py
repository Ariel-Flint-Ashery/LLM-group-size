#%%
import openai
import time
import yaml
from munch import munchify
import tiktoken
import numpy as np
import utils as ut

#%%
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)

# Set temperature to 0 for deterministic outcomes
temperature = config.params.temperature if hasattr(config.params, 'temperature') else 0

if temperature == 0:
    llm_params = {
        "max_tokens": 5,
        "logprobs": True,
        "top_logprobs": 5,
        "temperature": 0
    }
else:
    llm_params = {
        "temperature": temperature,
        "max_tokens": 5,
        "logprobs": True,
        "top_logprobs": 5
    }

meta_params = {
        "temperature": temperature,
        "max_tokens": 10,
        "logprobs": True,
        "top_logprobs": 5
    }

#%%
# Initialize OpenAI client
openai.api_key = config.model.API_TOKEN
client = openai.OpenAI(api_key=config.model.API_TOKEN)

# Get the model name from config (e.g., "gpt-4o", "gpt-3.5-turbo", etc.)
model_name = config.model.model_name

# Initialize tokenizer for the specific model
try:
    tokenizer = tiktoken.encoding_for_model(model_name)
except KeyError:
    # Fallback to cl100k_base encoding for newer models
    tokenizer = tiktoken.get_encoding("cl100k_base")

prepend_tokenized = tokenizer.encode("{'value':")
prepend_string_tokens = [tokenizer.decode([token]) for token in prepend_tokenized]
def query(payload, params=llm_params):
    """Make a request to OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=payload,
            **params
        )
        return response.choices[0]
    except Exception as e:
        print(f"API Error: {e}")
        return None

def get_meta_response(chat):
    overloaded = 1
    max_retries = 5
    retries = 0
    while overloaded == 1:
        response = query(chat, params=meta_params)
        message = response.message.content
        print(message, flush = True)
        if response is None:
            print('CAUGHT API ERROR')
            continue
        
        # Handle rate limiting
        if hasattr(response, 'error'):
            print("AN EXCEPTION: ", response.error)
            time.sleep(2.5)
            if "rate limit" in str(response.error).lower():
                print("RATE LIMIT REACHED")
                time.sleep(60)  # Wait 1 minute for OpenAI rate limits
            continue
        #print("Response content:", response.message.content)
        if "{'value':" not in response.message.content:
            retries += 1
            if retries > max_retries:
                print("Max retries exceeded. Exiting.")
                raise Exception("Response does not contain expected format")
            continue 
        else:
            overloaded = 0
    return message

def API_hit(chat, options, first_target_phrase):
    """Generate a response from the model."""
    
    overloaded = 1
    max_retries = 5
    retries = 0
    while overloaded == 1:
        response = query(chat)
        
        if response is None:
            print('CAUGHT API ERROR')
            continue
        
        # Handle rate limiting
        if hasattr(response, 'error'):
            print("AN EXCEPTION: ", response.error)
            time.sleep(2.5)
            if "rate limit" in str(response.error).lower():
                print("RATE LIMIT REACHED")
                time.sleep(60)  # Wait 1 minute for OpenAI rate limits
            continue
        #print("Response content:", response.message.content)
        if "{'value':" not in response.message.content:
            retries += 1
            if retries > max_retries:
                print("Max retries exceeded. Exiting.")
                raise Exception("Response does not contain expected format")
            continue 
            
        try:
            # Extract logprobs from OpenAI response
            if response.logprobs and response.logprobs.content:
                outputs = []
                token_outputs = []
                
                for token_logprob in response.logprobs.content:
                    # Get top logprobs for each token position
                    top_logprobs = token_logprob.top_logprobs if token_logprob.top_logprobs else []
                    outputs.append(top_logprobs)
                    
                    # Extract token strings
                    tokens = [logprob.token for logprob in top_logprobs]
                    token_outputs.append(tokens)
                
                if all(t in token_outputs for t in prepend_string_tokens) != True:
                    continue

            else:
                print("No logprobs returned")
                continue
                
        except Exception as e:
            print(f"Error processing logprobs: {e}")
            continue
        
        # Check if any of the first target phrases appear in the first position of any token
        if len([i for i in range(len(token_outputs)) if any(phrase == token_outputs[i][0] for phrase in first_target_phrase)]) != 0:
            overloaded = 0
        else:
            print("FIRST PHRASE NOT FOUND IN index 0 POSITION IN ANY TOKEN POSITION")
            print(f"Response: {response.message.content}")
            print("Output tokens:")
            for o in token_outputs:
                print(o, first_target_phrase, any(phrase == o for phrase in first_target_phrase))

    return [response, outputs, token_outputs]

def encode_decode_options(options):
    """Encode and decode options to get target phrases."""
    target_phrase = []
    
    for option in options:
        # Encode the option to get token IDs
        token_ids = tokenizer.encode(option)
        # Decode each token ID individually to get the token strings
        tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        target_phrase.append(tokens)
    
    first_target_phrase = [target[0] for target in target_phrase]
    print(f"Options tokens: {target_phrase}")
    print(f"First options tokens: {first_target_phrase}")
    
    first_target_id_dict = {option: first_target_phrase[i] for i, option in enumerate(options)}
    return first_target_id_dict

def get_probability_dict(options, prompt, first_target_id_dict, temperature=None, epsilon=np.finfo(float).eps):
    """Extract probabilities for target options from model response."""
    if temperature is None:
        temperature = config.params.temperature if hasattr(config.params, 'temperature') else 0
    
    first_target_phrase = [first_target_id_dict[option] for option in options]
    response, outputs, token_outputs = API_hit(chat=prompt, options=options, first_target_phrase=first_target_phrase)
    
    probability_dict = {opt: -np.inf for opt in options}

    # Find the token location where target options exist
    index_list = [i for i in range(len(token_outputs)) if all(phrase in token_outputs[i] for phrase in first_target_phrase)]

    if len(index_list) == 0:
        # Find any position where at least one target phrase appears at index 0
        matching_indices = [i for i in range(len(token_outputs)) if any(phrase == token_outputs[i][0] for phrase in first_target_phrase)]
        if matching_indices:
            index = matching_indices[0]
            # Find the winning option
            for idx, phrase in enumerate(first_target_phrase):
                if token_outputs[index][0] == phrase:
                    selected_option = options[idx]
                    # Get the logprob for the first token at this position
                    winning_prob = outputs[index][0].logprob if outputs[index] else 0.0
                    probability_dict[selected_option] = winning_prob
                    break
    else:
        # All target phrases appear in the same token position
        token_position = index_list[0]
        
        for i, phrase in enumerate(first_target_phrase):
            try:
                # Find the index of the phrase in the token outputs
                phrase_index = token_outputs[token_position].index(phrase)
                selected_option = options[i]
                
                # Get the logprob for this token
                # Note: OpenAI logprobs are already log probabilities, not raw logits
                # They represent log P(token | context) and are normalized over the vocabulary
                winning_prob = outputs[token_position][phrase_index].logprob
                probability_dict[selected_option] = max(winning_prob, np.log(epsilon))
                
            except (ValueError, IndexError):
                # Token not found in top logprobs
                continue

    # Renormalize probabilities
    options_log_probs = list(probability_dict.values())
    
    if -np.inf in options_log_probs:
        # Convert to regular probabilities, normalize, then back to log
        probs = np.exp(options_log_probs)
        normed_probs = ut.normalize_probs(probs)
        normed_log_probs = np.log(normed_probs)
    else:
        # Use utility function to normalize log probabilities
        normed_log_probs = ut.normalize_logprobs(options_log_probs)

    # Update probability dictionary with normalized values
    for option, log_prob in zip(probability_dict.keys(), normed_log_probs):
        probability_dict[option] = log_prob
    # print(f"Normalized log probs: {normed_log_probs}")
    return probability_dict
