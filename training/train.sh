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
    --batch_size 64 \
    --learning_rate 3e-4 \
    --val_set_size 0 \
    --use_chat_prompt \
    --train_on_inputs False \
