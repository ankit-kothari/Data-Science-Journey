#pip install poetry
#poetry install

#activate the poetry enviorment
#source $(poetry env info --path)/bin/activate
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#kick off the dataloader script
#torchrun --standalone --nnodes 1 --nproc_per_node 8 00_dataloader_script.py   --model Qwen/Qwen3-4B   --data toy_corpus_large.txt   --seq_len 2048   --global_bs 8   --num_inspect 2



#torchrun --standalone --nnodes 1 --nproc_per_node 4 01_DDP_baseline_OOM.py \
  --model Qwen/Qwen3-1.7B \
  --data toy_corpus_large.txt --seq_len 2048 --global_bs 4 --lr 2e-4



#torchrun --standalone --nnodes 1 --nproc_per_node 4 02_FSDP_ZeRO2_Strategy.py   --model Qwen/Qwen3-4B   --data toy_corpus_large.txt   --seq_len 2048   --global_bs 4   --lr 2e-4   --steps 5


#torchrun --standalone --nnodes 1 --nproc_per_node 4 03_FSDP_ZeRO3_Strategy.py   --model Qwen/Qwen3-4B   --data toy_corpus_large.txt   --seq_len 2048   --global_bs 4   --lr 2e-4   --steps 5


#git config --global user.email "ankit256@gmail.com"
#git config --global user.name ankit-kothari

