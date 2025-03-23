import os
import time
import math
import pickle
import tiktoken
from bpemb import BPEmb
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from transformers import load_tf_weights_in_gpt2
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
from transformers import GPT2Tokenizer

from model import GPTConfig, GPT

n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
bias = False
device = 'cuda'
init_from = 'scratch' # can be scratch or from checkpoint
dropout = 0 # 0 for pretraining and >0.1 for finetuning and other tasks
RPE = False # True if Rotational Positional Embedding is required false if not required 
SWE = False  # True if Sliding Window Attention is required false if not required
GQA = True

# attempt to derive vocab_size from the dataset
# meta_path = os.path.join(data_dir, 'meta.pkl')
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']


model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50257, dropout=dropout, rpe = RPE, swe = SWE, gqa = GQA) # start with model_args from command line

meta_vocab_size = None
if init_from == 'scratch':
    print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'checkpoint':
    # resume training from a checkpoint.
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    # print(dir(model))
    print (model.__getattr__)
    ckpt_path = r"models\124M"

    
    load_tf_weights_in_gpt2(model, gptconf, ckpt_path)

    pytorch_weights_dump_path = "pytorch_model" + "/" + WEIGHTS_NAME
    pytorch_config_dump_path = "pytorch_model" + "/" + CONFIG_NAME
    print(f"Save PyTorch model to {pytorch_weights_dump_path}")
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print(f"Save configuration file to {pytorch_config_dump_path}")
    with open(pytorch_config_dump_path, "w", encoding="utf-8") as f:
        f.write(gptconf.to_json_string())

    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model.to(device)

print (model)
# print (model.parameters)

# enc = tiktoken.get_encoder("gpt2")
# enc = BPEmb(lang="en", vs=25000, dim=300)
# sample_data = "This is my submission for Contlo round 2"


# sample_ids = enc.encode_ids(sample_data)
# sample_ids = torch.tensor(sample_ids)
# print (sample_ids)

# output = model(sample_ids)
# print (output)
model.eval()
num_samples = 1
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8
top_k = 200
start = "Hi my name is udit agarwal, final year student at IIT JodhpurHi my name is udit agarwal, final year student at IIT Jodhpur"
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')