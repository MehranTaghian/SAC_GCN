CUDA_VISIBLE_DEVICES=0 python main.py --env-name FetchReachEnv-v0 --exp-type test --automatic_entropy_tuning True --num_steps 300000 --start_steps 5000 -dsf 100 --cuda --seed 0

CUDA_VISIBLE_DEVICES=0 python main.py --env-name FetchReachEnv-v2 --exp-type test --automatic_entropy_tuning True --num_steps 300000 --start_steps 5000 -dsf 100 --cuda --seed 0

CUDA_VISIBLE_DEVICES=0 python main.py --env-name FetchReachEnv-v4 --exp-type test --automatic_entropy_tuning True --num_steps 300000 --start_steps 5000 -dsf 100 --cuda --seed 0

CUDA_VISIBLE_DEVICES=0 python main.py --env-name AntEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 10000000 --start_steps  10000 -dsf 10 --cuda --seed 0 --aggregation sum

CUDA_VISIBLE_DEVICES=0 python main.py --env-name AntEnv-v0 --exp-type standard --automatic_entropy_tuning True --num_steps 10000000 --start_steps  10000 -dsf 10 --cuda --seed 0

CUDA_VISIBLE_DEVICES=0 python main.py --env-name HalfCheetahEnv-v0 --automatic_entropy_tuning True --num_steps 10000000 --start_steps  10000 -dsf 10 --cuda --seed 0

