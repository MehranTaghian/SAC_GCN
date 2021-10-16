MAIN_FILE="/home/taghianj/Documents/SAC_GCN/main.py"

#python main.py --env-name "FetchReachEnv-v0" --automatic_entropy_tuning True --num_steps 10000 --cuda

tmux new-session -d -s sacgcn-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 1000000 --start_steps 500 --cuda --seed 0"
