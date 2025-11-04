#%%
import yaml
from munch import munchify
with open("config.yaml", "r") as f:
    doc = yaml.safe_load(f)
config = munchify(doc)
import huggingface_hub
huggingface_hub.login(config.model.API_TOKEN)
print('Start', flush=True)
import sys
import torch
print(f'torch available: {torch.cuda.is_available()}', flush=True)
print(torch.version.cuda)
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i))
from torch import bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
print(f'torch available: {torch.cuda.is_available()}', flush=True)
if "gemma-3" not in config.model.model_name:
    import bitsandbytes
    print(f'bitsandbytes: {bitsandbytes.__version__}', flush=True)
import accelerate
print(f'python: {sys.version}', flush=True)
print(f'torch: {torch.__version__}', flush=True)
print(f'transformers: {transformers.__version__}', flush=True)

print(f'accelerate: {accelerate.__version__}', flush=True)
model_name = config.model.model_name
print(f'model: {model_name}')
import numpy as np
import gc
import utils as ut
import torch.nn.functional as F
# %%

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

flush()
quantized = config.model.quantized
# loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, token=config.model.API_TOKEN)

if not quantized:
    # full precision model
    print('Loading full precision model', flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True,token=config.model.API_TOKEN, cache_dir = 'MODEL_FOLDER')
    model = model.to('cuda')
    model.config.use_cache = False
else:
    # quantized model
    print('Loading quantized model', flush=True)
    bnb_config = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, resume_download = True, device_map='cuda:0', quantization_config=bnb_config, cache_dir = 'MODEL_FOLDER')
    model.config.use_cache = False

def base_query(text, temperature = config.params.temperature, max_new_tokens = 6, options = None):
    if config.model.chat_template_is_avail:
        inputs = tokenizer.apply_chat_template(text, return_tensors="pt", continue_final_message=True).to("cuda:0")
    else:
        inputs = tokenizer.encode(text, return_tensors = "pt").to("cuda:0")

    with torch.no_grad():
        if temperature == 0:
            outputs = model.generate(inputs, max_new_tokens = 5,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = False, pad_token_id=tokenizer.eos_token_id)
        else:
            outputs = model.generate(inputs, max_new_tokens = max_new_tokens,output_scores=True, return_dict_in_generate=True, output_hidden_states=True, do_sample = True, temperature = temperature, pad_token_id=tokenizer.eos_token_id)
    return inputs, outputs

def query(text, temperature = config.params.temperature, max_new_tokens = 5, options = None):
    inputs, outputs = base_query(text = text, temperature=temperature, max_new_tokens=max_new_tokens, options = options)    

    generated_tokens = outputs.sequences[0]
    prompt_length = inputs.shape[1]
    generated_text = tokenizer.decode(generated_tokens[prompt_length:], skip_special_tokens=True)

    return {'generated_text': generated_text, 'generated_tokens': generated_tokens[prompt_length:]}

def get_meta_response(chat):
    """Generate a response from the model."""
    overloaded = 1
    response = query(text = chat, max_new_tokens=8)
    print(response['generated_text'], flush=True)
    return response['generated_text']

def encode_decode_options(options):
    target_encodings = [tokenizer.encode(option, add_special_tokens=False) for option in options]
    target_phrase = [[tokenizer.decode(target_token_id, skip_special_tokens=True) for target_token_id in encoding] for encoding in target_encodings]
    first_target_phrase = [target[0] for target in target_phrase]
    first_target_encoding = [target[0] for target in target_encodings]
    first_target_id_dict = {option: first_target_encoding[i] for i, option in enumerate(options)}
    return first_target_id_dict

def get_probability_dict(options, prompt, first_target_id_dict, temperature = config.params.temperature, epsilon=np.finfo(float).eps):
    # Compute transition scores (fix: pass full list of score tensors)
    inputs, outputs = base_query(text = prompt, temperature=temperature, max_new_tokens=5, options = options)    

    generated_tokens = outputs.sequences[0]
    # generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    first_target_encoding = [first_target_id_dict[option] for option in options]
    probability_dict = {opt: -np.inf for opt in options}
    options_log_probs = []
    # find generation probability of first token in each action label. Make sure that these tokens are different for the possible action labels!
    for choice in first_target_encoding:
        logits = outputs.scores[0]
        probabilities = torch.log_softmax(logits, dim=-1)
        # impose machine precision floor
        target_log_prob = max(np.log(epsilon), probabilities[0,  choice].item())
        options_log_probs.append(target_log_prob)
    # normalize log prob over all choice probabilities for this configuration and prompt
    if -np.inf in options_log_probs:
        normed_probs = ut.normalize_probs(np.exp(options_log_probs))
        normed_log_probs = np.log(normed_probs)
    else:
        normed_log_probs = ut.normalize_logprobs(options_log_probs)
    
    for option, prob in zip(options, normed_log_probs):
        probability_dict[option] = prob
    return probability_dict