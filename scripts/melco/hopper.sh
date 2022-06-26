MAIN_FILE="/local/melco3/taghianj/SAC_GCN/Controller/basic/main.py"

# standard
#tmux new-session -d -s hopper-0 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s hopper-1 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s hopper-2 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s hopper-3 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s hopper-4 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s hopper-5 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s hopper-6 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s hopper-7 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s hopper-8 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s hopper-9 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type standard --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"

# thigh
#tmux new-session -d -s thigh-0 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s thigh-1 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s thigh-2 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s thigh-3 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s thigh-4 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s thigh-5 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s thigh-6 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s thigh-7 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s thigh-8 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s thigh-9 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"
tmux new-session -d -s thigh-10 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 10"
tmux new-session -d -s thigh-11 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 11"
tmux new-session -d -s thigh-12 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 12"
tmux new-session -d -s thigh-13 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 13"
tmux new-session -d -s thigh-14 "CUDA_VISIBLE_DEVICES=0 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 14"
tmux new-session -d -s thigh-15 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 15"
tmux new-session -d -s thigh-16 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 16"
tmux new-session -d -s thigh-17 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 17"
tmux new-session -d -s thigh-18 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 18"
tmux new-session -d -s thigh-19 "CUDA_VISIBLE_DEVICES=1 python $MAIN_FILE --env-name Hopper-v2 --exp-type thigh_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 19"

# leg
#tmux new-session -d -s leg-0 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s leg-1 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s leg-2 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s leg-3 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s leg-4 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s leg-5 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s leg-6 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s leg-7 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s leg-8 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s leg-9 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"
tmux new-session -d -s leg-10 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 10"
tmux new-session -d -s leg-11 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 11"
tmux new-session -d -s leg-12 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 12"
tmux new-session -d -s leg-13 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 13"
tmux new-session -d -s leg-14 "CUDA_VISIBLE_DEVICES=3 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 14"
tmux new-session -d -s leg-15 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 15"
tmux new-session -d -s leg-16 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 16"
tmux new-session -d -s leg-17 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 17"
tmux new-session -d -s leg-18 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 18"
tmux new-session -d -s leg-19 "CUDA_VISIBLE_DEVICES=4 python $MAIN_FILE --env-name Hopper-v2 --exp-type leg_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 19"

# foot
#tmux new-session -d -s foot-0 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 0"
#tmux new-session -d -s foot-1 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 1"
#tmux new-session -d -s foot-2 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 2"
#tmux new-session -d -s foot-3 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 3"
#tmux new-session -d -s foot-4 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 4"
#tmux new-session -d -s foot-5 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 5"
#tmux new-session -d -s foot-6 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 6"
#tmux new-session -d -s foot-7 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 7"
#tmux new-session -d -s foot-8 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 8"
#tmux new-session -d -s foot-9 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 9"
tmux new-session -d -s foot-10 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 10"
tmux new-session -d -s foot-11 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 11"
tmux new-session -d -s foot-12 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 12"
tmux new-session -d -s foot-13 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 13"
tmux new-session -d -s foot-14 "CUDA_VISIBLE_DEVICES=6 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 14"
tmux new-session -d -s foot-15 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 15"
tmux new-session -d -s foot-16 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 16"
tmux new-session -d -s foot-17 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 17"
tmux new-session -d -s foot-18 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 18"
tmux new-session -d -s foot-19 "CUDA_VISIBLE_DEVICES=7 python $MAIN_FILE --env-name Hopper-v2 --exp-type foot_joint --automatic_entropy_tuning True --num_episodes 10000 --start_steps 10000 -dsf 100 --cuda --seed 19"
