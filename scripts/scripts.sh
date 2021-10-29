MAIN_FILE="/home/taghianj/Documents/SAC_GCN/main.py"

#python main.py --env-name "FetchReachEnv-v0" --automatic_entropy_tuning True --num_steps 10000 --cuda

#tmux new-session -d -s sacgcn-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 1000000 --start_steps 5000 --cuda --seed 0"

CUDA_VISIBLE_DEVICES=0 python main.py --env-name FetchReachEnv-v0 --automatic_entropy_tuning True --num_steps 2000000 --start_steps 5000 -msf 10 --cuda --seed 0

CUDA_VISIBLE_DEVICES=0 python main.py --env-name AntEnv-v0 --automatic_entropy_tuning True --num_steps 2000000 --start_steps 5000 -msf 10 --cuda --seed 0


#Evaluation
python evaluate.py --env-name FetchReachEnv-v0 -chp ~/Documents/SAC_GCN/checkpoints/sac_checkpoint_FetchReachEnv-v0_