#CUDA_VISIBLE_DEVICES=0 python main.py --env-name FetchReachEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 200000 --start_steps 5000 -dsf 1000 --cuda --seed 0

#CUDA_VISIBLE_DEVICES=0 python main.py --env-name AntEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 10000000 --start_steps  20000 -dsf 10 --cuda --seed 0
