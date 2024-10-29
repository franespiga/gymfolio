import argparse
import sys
import logging
import json
import pandas as pd
import datetime
from stable_baselines3 import PPO, SAC, DDPG, DQN
import os
import numpy as np
import torch
from tqdm import tqdm

sys.path.append('../src')

from src.envs.base import PortfolioOptimizationEnv
from src.envs.common import create_return_matrices, decompose_weights_tensor

logger = logging.getLogger('StableBaselines3_example')
logger.setLevel(logging.DEBUG)


def update_logger(stage: str, logdir: str):
    logging.basicConfig(filename=f'{logdir}/{stage}.log', encoding='utf-8', level=logging.WARNING,
                        format='%(asctime)s %(message)s')

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """
    # I/O args
    parser = argparse.ArgumentParser(usage='Trains a Stable Baselines 3 agent')
    parser.add_argument('--input-path', help='Input HDF5 file', type=str, default='../data')
    parser.add_argument('--output-path', help='Output folder to store results', type=str, default=None)

    # Environment args
    parser.add_argument('--use-cash-instrument', help='Use CASH instrument as non-risk position', type=bool,
                        default=True)
    parser.add_argument('--lookback', help='Lookback window', type=int, default=16)
    parser.add_argument('--rebalance-every', help='Periods to rebalance positions', type=int, default=5)
    parser.add_argument('--continuous-weights', help='Consider continuous weights for rabalancing', type=bool,
                        default=False)
    parser.add_argument('--convert-to-terminated-truncated', help='Gym legacy (DONE) or new (TRUNCATED/TERMINATED)',
                        type=bool, default=True)
    parser.add_argument('--use-prices-as-indicators', help = "Add PRICES as indicators", type = bool, default = True)

    # Agent args
    parser.add_argument('--agent-algorithm', help="SB3 agent algorithm", type=str, default='PPO')

    # Training args
    parser.add_argument('--reference-date', help='Reference date for TRAIN/Test split', type=str, default='2016-01-01')
    parser.add_argument('--start-date', help='Start date for the study', type=str, default='2000-01-01')
    parser.add_argument('--timesteps', help='Training timesteps', type=int, default=1_500)
    parser.add_argument('--max-trajectory-len', help='Episode length (periods)', type=int, default=252)
    parser.add_argument('--train-model', help='Train model', type=bool, default=True)

    # Other args
    parser.add_argument('--debug', help='Debug indicators and intermediate storage', type=bool, default=False)
    parser.add_argument('--verbose', help='Verbose', type=int, default=1)

    args = vars(parser.parse_args())
    return args


if __name__ == "__main__":
    args = get_arguments()
    input_path = args['input_path']

    lookback = args['lookback']
    rebalance_every = args['rebalance_every']
    agent_algorithm = args['agent_algorithm']

    study_name = f"SB3_example"
    output_path = f"{args['output_path'] if args['output_path'] is not None else os.path.join(input_path,study_name)}"

    log_dir = os.path.join(output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = update_logger("sb3_agents", log_dir)

    logger.info("Script args")
    for k,v in args.items():
        logger.info(f"{k} : {v} - {type(v)}")

    # DATA PREPARATION ---
    logger.info("Loading datasets")

    try:
        data = pd.read_hdf(f"{input_path}/example.h5", 'instruments')
        data.index = pd.to_datetime(data.index).strftime('%Y-%m-%d')
        data = data[args['start_date']:]
        instruments = [i[0] for i in data.columns if i[1]=='Close']
    except:
        logger.error(f"Error reading HDF5 instrument dataset on {input_path}")
        raise ValueError("No Instruments found")

    try:
        indicators = pd.read_hdf(f"{input_path}/example.h5", 'technical_indicators')
        indicators.index = pd.to_datetime(indicators.index).strftime('%Y-%m-%d')
    except:
        logger.error(f"Error reading HDF5 technical indicators dataset on {input_path}, not found")


    if os.path.isfile(f"{input_path}/metadata.json"):
        with open(f"{input_path}/metadata.json", 'r') as reader:
            metadata = json.load(reader)
    else:
        metadata = {}

    metadata['agent_algorithm'] = agent_algorithm
    metadata['lookback'] = int(lookback)
    metadata['rebalance_every'] = int(rebalance_every)
    metadata['timesteps'] = args['timesteps']

    logger.info(f"Preparing TRAIN/Test split to/fro {args['reference_date']}")
    train_data = data[:args['reference_date']]
    test_data = data[args['reference_date']:]

    train_indicators = indicators[:args['reference_date']]
    test_indicators = indicators[args['reference_date']:]

    if args['use_prices_as_indicators']:
        train_indicators = train_indicators.merge(train_data, right_index = True, left_index = True)
        test_indicators = test_indicators.merge(test_data, right_index = True, left_index = True)

    # ENVIRONMENT DEFINITION ---
    logger.info("Preparing environment")
    agent_types = {
        'SAC': 'continuous',
        'A2C': 'continuous',
        'DDPG': 'continuous',
        'DQN': 'discrete',
        'PPO': 'discrete'}

    logger.warning(f"Action order: {instruments}")
    env = PortfolioOptimizationEnv(
        df_ohlc=train_data.dropna(),
        df_observations=train_indicators.dropna(),
        rebalance_every=rebalance_every,
        max_trajectory_len=args['max_trajectory_len'],
        observation_frame_lookback=lookback - 1,
        continuous_weights=args['continuous_weights'],
        verbose=args['verbose'],
        agent_type=agent_types[agent_algorithm],
        render_mode='tile',
        convert_to_terminated_truncated=args['convert_to_terminated_truncated']
    )

    # AGENT INITIALIZATION ---
    logger.info("Initializing agent")

    if agent_algorithm == 'SAC':
        model = SAC('MlpPolicy', env,verbose=1)
    elif agent_algorithm == 'DQN':
        model = DQN('MlpPolicy', env,verbose=1)
    elif agent_algorithm == 'PPO':
        model = PPO('MlpPolicy', env,verbose=1)
    elif agent_algorithm == 'DDPG':
        model = DDPG('MlpPolicy', env, verbose=1)
    else:
        raise ValueError(f"Algorithm {agent_algorithm} is not supported")

    if args['train_model']:
        logger.info("Starting training")
        model.learn(total_timesteps=args['timesteps'])
        model.save(f'{output_path}/models/final_model')
    else:
        logger.warning(f"Skipping training and reloading model from {output_path}")

    models = {
        'FINAL':f'{output_path}/models/final_model.zip',
    }

    os.makedirs(f"{output_path}/results", exist_ok=True)
    eval_data = pd.concat([train_indicators, test_indicators], axis = 0)
    logger.info(f"Evaluation data shape {eval_data.shape}")
    for model_key, model_path in models.items():
        model.load(model_path)

        logger.info(f"Evaluating {model_key} model on Test period from {args['reference_date']} to {max(test_data.index)}")

        obs, info = env.reset()
        decision_series = {}
        action = None

        different_actions = []
        for idx, row in eval_data.dropna().tail(eval_data.dropna().shape[0] - lookback).iterrows():
            if action is not None:
                if agent_types[agent_algorithm]=='discrete':
                    actions = np.zeros(len(instruments))
                    actions[int(action[0])] = 1.0
                    different_actions.append(int(action[0]))
                    decision_series[idx] = actions
                else:
                    decision_series[idx] = action[0]
            observation_frame = eval_data.loc[:idx, :].tail(lookback)

            if observation_frame.shape[0] < lookback:
                break
            action = model.predict(torch.Tensor(observation_frame.values), deterministic=True)

        df_weights = pd.DataFrame.from_dict(decision_series, orient='index', columns=instruments)
        df_weights = df_weights.div(df_weights.sum(axis=1), axis=0)
        df_weights.index = pd.to_datetime(df_weights.index)
        df_weights.to_csv(f"{output_path}/results/weights_{model_key}.csv")

        data.index = pd.to_datetime(data.index)
        R_h, R_b, R_s = create_return_matrices(data)
        R_h.index = pd.to_datetime(R_h.index)
        R_b.index = pd.to_datetime(R_b.index)
        R_s.index = pd.to_datetime(R_s.index)

        missing_indices = set(df_weights.index)-set(data.index)
        logger.warning(f"Missing indices: {list(missing_indices)}")
        missing_indices = set(data.index)-set(df_weights.index)
        logger.warning(f"Missing indices: {list(missing_indices)}")

        current_weights = np.zeros(len(instruments))
        returns = {}
        for idx, row in tqdm(df_weights.iterrows()):
            if idx in R_h.index:
                new_weights = row.values
                w_h, w_b, w_s = decompose_weights_tensor(torch.Tensor(new_weights), torch.Tensor(current_weights))
                r_h = torch.Tensor(R_h.loc[idx])
                r_b = torch.Tensor(R_b.loc[idx])
                r_s = torch.Tensor(R_s.loc[idx])

                pf_h = torch.dot(w_h, r_h)
                pf_b = torch.dot(w_b, r_b)
                pf_s = torch.dot(w_s, r_s)

                pf_rets = pf_h + pf_b + pf_h

                returns[idx] = pf_rets.detach().numpy().tolist()

                current_weights = new_weights

        df_strategy = pd.DataFrame.from_dict(returns,
                                             orient = 'index',
                                             columns = ['strategy_returns'])
        df_strategy.index = pd.to_datetime(df_strategy.index)
        print(df_strategy)

        df_buy_and_hold = data.loc[:,[(i, 'Close') for i in instruments]].pct_change()
        df_buy_and_hold.columns = [i[0] for i in df_buy_and_hold.columns]
        df_buy_and_hold['equal_weights'] = df_buy_and_hold.mean(axis = 1)
        df_buy_and_hold.index = pd.to_datetime(df_buy_and_hold.index)
        print(df_buy_and_hold)

        df_comparison = df_buy_and_hold.merge(df_strategy, right_index = True, left_index = True)
        df_comparison.to_csv(f'{output_path}/results/strategy_comparison_{model_key}.csv')

        with open(f"{output_path}/metadata.json",'w') as file:
            json.dump(metadata, file)

