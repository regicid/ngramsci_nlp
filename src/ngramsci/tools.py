"""Model preparation helpers."""
import os
import shutil


def merge_qwen(adapter, out):
    """Merge a lattice-nlp coref LoRA into its base and save a standalone model for vLLM (bf16, CPU, ~20 GB RAM)."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftConfig, PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    base_id = PeftConfig.from_pretrained(adapter).base_model_name_or_path
    print("base:", base_id, "adapter:", adapter, flush=True)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(base_id, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")

    # processor / preprocessor files from the base (vLLM's Qwen3.5 loader expects them),
    # then tokenizer + chat template from the adapter repo (what the model was trained with)
    base_dir = snapshot_download(base_id, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"])
    for f in os.listdir(base_dir):
        dst = os.path.join(out, f)
        if f.endswith((".json", ".jinja", ".txt", ".model")) and f not in ("config.json", "model.safetensors.index.json", "generation_config.json") \
                and not os.path.exists(os.path.join(out, f)):
            shutil.copyfile(os.path.join(base_dir, f), dst)   # copyfile: no read-only perms from HF cache
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    print("saved", out, flush=True)
