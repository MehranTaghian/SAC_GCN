MAIN_FILE="/local/melco3/taghianj/SAC_GCN/Controller/graph/main.py"

tmux new-session -d -s halfcheetahgraphv0-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s halfcheetahgraphv0-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s halfcheetahgraphv0-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s halfcheetahgraphv0-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s halfcheetahgraphv0-4 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s halfcheetahgraphv0-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s halfcheetahgraphv0-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s halfcheetahgraphv0-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s halfcheetahgraphv0-8 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s halfcheetahgraphv0-9 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name HalfCheetahEnvGraph-v0 --exp-type newmodel --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

