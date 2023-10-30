# Model Training for On-Demand IE

### Local Setup

Install dependencies

```bash
pip install -r requirements.txt
```

### Training (`finetune.py`)

We train LLaMA and LoRA based models for the On-Demand IE model with the following usage:

```bash
export CUDA_VISIBLE_DEVICES=0,1
# export TRANSFORMERS_CACHE=/path/to/cache

WORLD_SIZE=2  torchrun --nproc_per_node=2 --master_port=12345 finetune.py \
    --base_model "elinas/llama-7b-hf-transformers-4.29" \
    --num_epochs 10 \
    --cutoff_len 2048 \
    --data_path "data/training_data.jsonl" \
    --output_dir "odie-7b" \
    --lora_target_modules "[q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj,embed_tokens,lm_head]" \
    --lora_r 16 \
    --micro_batch_size 4 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --val_set_size 0 \
    --use_chat_prompt \
    --train_on_inputs False \
```

### Inference (`inference.py`)

For the trained LoRA model, we use the following code to inference on the test set and store the model output in the `output` folder.

```bash
export CUDA_VISIBLE_DEVICES=0
# export TRANSFORMERS_CACHE=/path/to/cache

python inference.py \
    --base_model 'elinas/llama-7b-hf-transformers-4.29' \
    --lora_weights 'odie-7b' \
    --cot False \
```
