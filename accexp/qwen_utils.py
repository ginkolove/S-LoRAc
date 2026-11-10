"""Shared Qwen adapter preflight and native chat-template formatting."""

import hashlib
from pathlib import Path


def check_adapter(args, *, allow_template_mismatch=False):
    from peft import PeftConfig
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    adapter = PeftConfig.from_pretrained(args.lora, local_files_only=True)
    if config.model_type != "qwen2" or adapter.peft_type != "LORA":
        raise ValueError("Expected Qwen2/Qwen2.5 and a standard PEFT LoRA adapter")
    if (adapter.bias != "none" or adapter.use_dora or getattr(adapter, "lora_bias", False)
            or adapter.modules_to_save or adapter.rank_pattern or adapter.layers_to_transform is not None):
        raise ValueError("This launcher requires uniform-rank additive LoRA without modified base parameters")
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    widths = dict(q_proj=(config.hidden_size, config.num_attention_heads * head_dim),
        k_proj=(config.hidden_size, config.num_key_value_heads * head_dim),
        v_proj=(config.hidden_size, config.num_key_value_heads * head_dim),
        o_proj=(config.num_attention_heads * head_dim, config.hidden_size),
        up_proj=(config.hidden_size, config.intermediate_size),
        gate_proj=(config.hidden_size, config.intermediate_size),
        down_proj=(config.intermediate_size, config.hidden_size))
    if not isinstance(adapter.target_modules, (set, list)) or not set(adapter.target_modules) <= set(widths):
        raise ValueError("Unsupported or regex LoRA target modules")
    expected = {}
    for layer in range(config.num_hidden_layers):
        for module in adapter.target_modules:
            section = "self_attn" if module in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            name = f"base_model.model.model.layers.{layer}.{section}.{module}"
            inp, out = widths[module]
            expected[name + ".lora_A.weight"] = [adapter.r, inp]
            expected[name + ".lora_B.weight"] = [out, adapter.r]
    with safe_open(str(Path(args.lora) / "adapter_model.safetensors"), framework="pt", device="cpu") as weights:
        if set(weights.keys()) != set(expected):
            raise ValueError("Adapter tensors do not cover exactly the configured Qwen layers/targets")
        for name, shape in expected.items():
            if weights.get_slice(name).get_shape() != shape:
                raise ValueError(f"Adapter/model shape mismatch: {name}, expected {shape}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    template_equal = None
    adapter_template_hash = None
    if (Path(args.lora) / "tokenizer_config.json").exists():
        other = AutoTokenizer.from_pretrained(args.lora, local_files_only=True)
        if other.get_vocab() != tokenizer.get_vocab() or other.special_tokens_map != tokenizer.special_tokens_map:
            raise ValueError("Adapter and base vocabularies/special tokens differ; resolve before evaluating")
        template_equal = other.chat_template == tokenizer.chat_template
        if not template_equal and not allow_template_mismatch:
            raise ValueError("Adapter and base chat templates differ; resolve before evaluating")
        if other.chat_template is not None:
            adapter_template_hash = hashlib.sha256(other.chat_template.encode()).hexdigest()
    return dict(model_type=config.model_type, hidden_size=config.hidden_size,
        layers=config.num_hidden_layers, attention_heads=config.num_attention_heads,
        kv_heads=config.num_key_value_heads, head_dim=head_dim, rank=adapter.r,
        target_modules=sorted(adapter.target_modules), checked_tensors=len(expected),
        adapter_declared_base=adapter.base_model_name_or_path,
        tokenizer_policy="base model vocabulary, special tokens and native chat template for all three modes",
        chat_template_equal=template_equal, adapter_chat_template_sha256=adapter_template_hash,
        base_chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        template_warning="Adapter template differs; deliberately unused. Training prompt remains unverified."
                         if template_equal is False else None,
        adapter_config_sha256=hashlib.sha256((Path(args.lora) / "adapter_config.json").read_bytes()).hexdigest())


def chat_prompt_builders(tokenizer, system_prompt, user_formatter=None):
    system = [dict(role="system", content=system_prompt)]
    user_formatter = user_formatter or (lambda text: text)

    def prefix_builder(turns):
        messages = list(system)
        for question, answer in turns:
            messages.extend((dict(role="user", content=user_formatter(question)),
                             dict(role="assistant", content=answer)))
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    header = prefix_builder([])

    def query_builder(question):
        rendered = tokenizer.apply_chat_template(system + [dict(role="user", content=user_formatter(question))],
                                                  tokenize=False, add_generation_prompt=True)
        if not rendered.startswith(header):
            raise ValueError("Chat template does not preserve the system-prefix boundary")
        return rendered[len(header):]

    return prefix_builder, query_builder
