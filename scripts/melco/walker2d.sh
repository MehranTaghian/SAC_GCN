MAIN_FILE="/local/melco3/taghianj/SAC_GCN/Controller/basic/main.py"

tmux new-session -d -s walker-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s walker-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s walker-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s walker-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s walker-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s walker-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s walker-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s walker-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s walker-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s walker-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Walker2d-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

tmux new-session -d -s foot-0 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s foot-1 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s foot-2 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s foot-3 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s foot-4 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s foot-5 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s foot-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s foot-7 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s foot-8 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s foot-9 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Walker2d-v2 --exp-type foot_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

tmux new-session -d -s leg-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s leg-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s leg-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s leg-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s leg-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s leg-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s leg-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s leg-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s leg-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s leg-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type leg_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

tmux new-session -d -s thigh-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
tmux new-session -d -s thigh-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
tmux new-session -d -s thigh-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
tmux new-session -d -s thigh-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
tmux new-session -d -s thigh-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
tmux new-session -d -s thigh-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
tmux new-session -d -s thigh-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
tmux new-session -d -s thigh-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
tmux new-session -d -s thigh-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
tmux new-session -d -s thigh-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Walker2d-v2 --exp-type thigh_left_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"
