
cd /workspace/ColossalAI/applications/Chat/examples

# download model & data
python downloader.py

# torchrun --standalone --nproc_per_node=4 train_sft.py \
torchrun --standalone --nproc_per_node=4 train_sft.py \
    --pretrain "model_llama_7b" \
    --model 'llama' \
    --strategy colossalai_zero2 \
    --log_interval 10 \
    --save_path  /model_dir \
    --dataset dataset/alpaca_data_cn100000.json \
    --batch_size 4 \
    --accimulation_steps 8 \
    --lr 2e-5 \
    --max_datasets_size 512 \
    --max_epochs 1 \
