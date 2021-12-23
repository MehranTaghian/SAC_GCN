MAIN_FILE="/home/taghianj/Documents/SAC_GCN/main.py"

# FetchReachEnv-v0
#tmux new-session -d -s sacgcn-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 0"
tmux new-session -d -s sacgcn-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 1"
#tmux new-session -d -s sacgcn-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 2"
#tmux new-session -d -s sacgcn-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 3"
#tmux new-session -d -s sacgcn-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 4"
tmux new-session -d -s sacgcn-5 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 5"
tmux new-session -d -s sacgcn-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 6"
tmux new-session -d -s sacgcn-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 7"
tmux new-session -d -s sacgcn-8 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 8"
tmux new-session -d -s sacgcn-9 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 300000 --start_steps 10000 -msf 10 --cuda --seed 9"



#CUDA_VISIBLE_DEVICES=0 python main.py --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 2000000 --start_steps 5000 -msf 10 --cuda --seed 0

#CUDA_VISIBLE_DEVICES=0 python main.py --env-name AntEnv-v0 --automatic_entropy_tuning True --num_steps 10000000 --start_steps 5000 -msf 10 --cuda --seed 0


#Evaluation
#python evaluate.py --env-name FetchReachEnv-v0 -chp ~/Documents/SAC_GCN/checkpoints/sac_checkpoint_FetchReachEnv-v0_