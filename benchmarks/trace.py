import json
import numpy as np
from typing import Dict
from tqdm import tqdm
import random
from transformers import AutoTokenizer

class Request:
    def __init__(self, req_id, model_dir, adapter_dir, prompt, prompt_len, output_len, req_time):
        self.req_id = req_id
        self.model_dir = model_dir 
        self.adapter_dir = adapter_dir
        self.prompt = prompt
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.req_time = req_time

    
    def __repr__(self):
        return f"req_id={self.req_id}, " \
               f"model_dir={self.model_dir}, adapter_dir={self.adapter_dir}, " \
               f"prompt_len={self.prompt_len}, output_len={self.output_len}, " \
               f"req_time={self.req_time}"


def dummy_prompt(prompt_len):
    return "Hello " * prompt_len


def sample_adapter_indices(num_adapters, alpha, tot_req, adapter_distribution="power"):
    if adapter_distribution == "zipf":
        ranks = np.arange(1, num_adapters + 1, dtype=np.float64)
        probs = 1.0 / np.power(ranks, alpha)
        probs = probs / probs.sum()
        return np.random.choice(num_adapters, size=tot_req, p=probs)

    probs = np.random.power(alpha, tot_req)
    return np.minimum((probs * num_adapters).astype(int), num_adapters - 1)


def _tokenizer_len(tokenizer, text, add_special_tokens=False):
    return len(tokenizer(text, add_special_tokens=add_special_tokens).input_ids)


def _find_exact_prompt_token_id(tokenizer):
    candidate_texts = [" hello", " world", " test", " data", " token", " prompt", " sample"]
    for text in candidate_texts:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        probe_ids = token_ids * 16
        probe_text = tokenizer.decode(
            probe_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if _tokenizer_len(tokenizer, probe_text, add_special_tokens=False) == len(probe_ids):
            return token_ids[0]

    max_scan = min(getattr(tokenizer, "vocab_size", 32000), 32000)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    for token_id in range(max_scan):
        if token_id in special_ids:
            continue
        probe_ids = [token_id] * 16
        probe_text = tokenizer.decode(
            probe_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if probe_text and _tokenizer_len(tokenizer, probe_text, add_special_tokens=False) == len(probe_ids):
            return token_id
    raise RuntimeError("Unable to find a stable single-token prompt unit for exact synthetic prompts")


def build_exact_token_prompt(tokenizer, prompt_len, prompt_cache: Dict[int, str] = None):
    if prompt_len <= 0:
        return ""

    if prompt_cache is not None and prompt_len in prompt_cache:
        return prompt_cache[prompt_len]

    unit_token_id = _find_exact_prompt_token_id(tokenizer)
    target_len = int(prompt_len)
    empty_special_len = _tokenizer_len(tokenizer, "", add_special_tokens=True)
    raw_target_len = max(target_len - empty_special_len, 0)

    def _decode_repeated(raw_len):
        return tokenizer.decode(
            [unit_token_id] * raw_len,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    prompt = _decode_repeated(raw_target_len)
    actual_len = _tokenizer_len(tokenizer, prompt, add_special_tokens=True)
    if actual_len != target_len:
        matched_prompt = None
        for candidate_raw_len in range(max(raw_target_len - 8, 0), raw_target_len + 9):
            candidate_prompt = _decode_repeated(candidate_raw_len)
            candidate_len = _tokenizer_len(tokenizer, candidate_prompt, add_special_tokens=True)
            if candidate_len == target_len:
                matched_prompt = candidate_prompt
                break
        if matched_prompt is None:
            raise ValueError(
                f"Exact synthetic prompt generation failed: target_len={target_len}, actual_len={actual_len}"
            )
        prompt = matched_prompt

    if prompt_cache is not None:
        prompt_cache[prompt_len] = prompt
    return prompt


def generate_requests(num_adapters, alpha, req_rate, cv, duration,
                      input_range, output_range,
                      adapter_dirs, # (base_dir, adapter_dir)
                      seed=42,
                      fixed_input_len=None,
                      fixed_output_len=None,
                      logical_input_len=None,
                      exact_prompt_tokens=False,
                      tokenizer_name=None,
                      adapter_distribution="power"):
    np.random.seed(seed)

    tot_req = int(req_rate * duration)

    # generate adapter id
    ind = sample_adapter_indices(num_adapters, alpha, tot_req, adapter_distribution=adapter_distribution)

    # generate input output len
    if fixed_input_len is not None:
        input_lens = np.full(tot_req, int(fixed_input_len), dtype=np.int64)
    else:
        input_lens = np.random.randint(input_range[0], input_range[1], tot_req)
    if fixed_output_len is not None:
        output_lens = np.full(tot_req, int(fixed_output_len), dtype=np.int64)
    else:
        output_lens = np.random.randint(output_range[0], output_range[1], tot_req)

    tokenizer = None
    prompt_cache = {}
    exact_prompt_unit_token_id = None
    if exact_prompt_tokens:
        if tokenizer_name is None:
            tokenizer_name = adapter_dirs[0][0]
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # generate timestamp
    requests = []
    tic = 0
    shape = 1 / (cv * cv)
    scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
    intervals = np.random.gamma(shape, scale, tot_req)
    for i in range(tot_req):
        tic += intervals[i]
        actual_prompt_len = int(input_lens[i])
        req_prompt_len = int(logical_input_len) if logical_input_len is not None else actual_prompt_len
        if exact_prompt_tokens:
            prompt = build_exact_token_prompt(tokenizer, actual_prompt_len, prompt_cache)
        else:
            prompt = dummy_prompt(actual_prompt_len)
        requests.append(Request(
            i,
            adapter_dirs[ind[i]][0],
            adapter_dirs[ind[i]][1],
            prompt,
            req_prompt_len,
            int(output_lens[i]),
            tic,
        ))
    return requests

def get_real_requests(trace_file, req_rate, duration, base_model, adapter_dirs, input_range, output_range, seed=42):
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    conversations = downsample(trace_file, req_rate, duration, tokenizer, input_range, output_range)
    model_mapping = generate_model_mapping(conversations, adapter_dirs)
    conversations = sort_and_rescale_by_req_time(conversations, duration)
    reqs = parse_into_req(base_model, conversations, model_mapping, tokenizer)
    return model_mapping.values(), reqs

# functions below are used to generate real requests
def downsample(json_file, req_rate, duration, tokenizer, input_range, output_range):
    with open(json_file, "r") as file:
       all_conversations = json.load(file)
    
    more_ratio = 2
    need_num = int(req_rate * duration)
    # sample a bit more than needed
    selected_indicies = np.random.choice(len(all_conversations), more_ratio * need_num, replace=False)
    downsampled_conversations = [all_conversations[idx] for idx in selected_indicies]
    for idx, conv in enumerate(downsampled_conversations):
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        if prompt_len >= input_range[1] or output_len >= output_range[1]:
            # to avoid OOM in some configurations
            downsampled_conversations.pop(idx)
    downsampled_conversations = downsampled_conversations[:need_num]
    print(f"Downsampled {len(downsampled_conversations)}")
    return downsampled_conversations 

def generate_model_mapping(conversations, adapter_dirs):
    model_mapping = {}
    num_ranks = [0] * len(adapter_dirs)
    for conv in conversations:
        model = conv["model"]
        if model not in model_mapping.keys():
            adapter_dir = random.choice(adapter_dirs)
            name = f"{adapter_dir}-{num_ranks[adapter_dirs.index(adapter_dir)]}"
            num_ranks[adapter_dirs.index(adapter_dir)] += 1
            model_mapping[model] = name
    print(model_mapping)
    return model_mapping

def sort_and_rescale_by_req_time(conversations, duration):
    # sort first
    sorted_conversations = sorted(conversations, key=lambda d: d['tstamp']) 
    interval_start = sorted_conversations[0]["tstamp"]
    interval_end = sorted_conversations[-1]["tstamp"]
    # print(f"sorted time step: {[s['tstamp'] for s in sorted_conversations]}")

    for conv in conversations:
        tstamp = conv["tstamp"]
        assert interval_start <= tstamp and tstamp <= interval_end
        rescaled_tstamp = (tstamp - interval_start) / (interval_end - interval_start) * duration
        conv["tstamp"] = rescaled_tstamp
    return sorted_conversations 

def parse_into_req(base_model, conversations, model_mapping, tokenizer):
    reqs = []
    for idx, conv in enumerate(tqdm(conversations, desc="parse into reqs")):
        model = conv["model"]
        name = model_mapping[model]
        # print(conv["conversation"][0]["content"])
        prompt_len = len(tokenizer(conv["conversation"][0]["content"]).input_ids)
        output_len = len(tokenizer(conv["conversation"][1]["content"]).input_ids)
        
        req = Request(req_id=idx, model_dir=base_model, adapter_dir=name, 
              prompt=conv["conversation"][0]["content"], prompt_len=prompt_len,
              output_len=output_len, req_time=conv["tstamp"])
        reqs.append(req)
    # print(reqs)
    return reqs
