# GymFolio

Welcome to GymFolio, a specialized Reinforcement Learning environment designed for the dynamic field of portfolio optimization. 
GymFolio allows practitioners to create investment strategies for financial investment using Reinforcement Learning. 

Built on Python, this environment provides a framework for training and evaluating various deep reinforcement 
learning models like PPO, SAC, and DQN in the context of managing and optimizing financial portfolios. It is compatible
with most of the agents in the package [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) and has
been tested with `DQN`, `DDPG`, `PPO` and `SAC`. 




## Environments

### Base environment

`PortfolioOptimizationEnv` is a base environment built using OHLC data and an arbitrary number of technical indicators.



The reward is simply the return window. It is located in `envs/base`.  
More sophisticated reward functions are in the children environments located in `envs/custom`, which have the *Sharpe Ratio* (`SharpeEnv`),
*Sortino Ratio* (`SortinoEnv`), *Calmar Ratio* (`CalmarEnv`) and *Tracking error* (`TrackingErrorEnv`).



#### Parameters
 * df_ohlc: `pandas Dataframe` with the Open, Close, High and Low prices at the desired granularity.
 * df_observations: `pandas Dataframe` with all the information used as observation of the environment. 
 * rebalance_every: `int, default = 1` time between actions in consecutive time units in the series. `rebalance_every = 4`, for instance, would mean that every 4 available dates a new rabalancing decision will be made.
A value of 1 is for continuous (daily in the default time units) rebalancing.
 * slippage: `float, default = 0.005`, price changes from the submission of the order to its fulfillment.
 * transaction_costs: `float, default = 0.002`, costs of placing an order.
 * continuous_weights: `bool, default = False`, should we consider the weights as continuous, based only on Close to Close price timeseries 
or split them in *Holding weights*, *Buying weights* and *Selling weighs* (More details in a follow-up section)
 * max_trajectory_len: `int, default = 252`, maximum time units for the trajectory.
 * return_observation_frame: `bool, default = False`, the environment can return a frame with the observations from all the timesteps between rebalancing dates, 
or alternatively, the most recent observation. 


#### Reward


#### Environment behavior
Let's assume that we have `rebalance_every=4`, `return_observation_frame=True` with continuous weights and a trajectory of 20.
The environment starts on `2023-03-06` and our first rebalancing date is `2023-03-13`, the next on the `2023-03-17`.

1. An observation frame with dates `2023-03-07`:`2023-03-13` is sliced from df_observations and return to the agent 
in the previous `env.step()`.
2. The agent process the observations and submits an action, the new portfolio weights. 
3. As the environment has continuous weights, no further processing is done to them. 
4. Return and observation frames are extracted between the next available date from the rebalancing date (`2023-03-14` for `2023-03-13`).
The observation will be returned by the current `env.step()`. 
5. The reward is computed as the weighed average return of each instrument for the time window between the effective date of the action (`2023-03-14`)
and the next rebalancing date (`2023-03-17`).
6. Date trackers are updated and the environment checks if the trajectory has the maximum length of 10, or if there are no more available rebalancing dates in the history. 
If that is the case, `env.step()` returns `done=True`, else the environment proceeds to the next interaction.



### Weight and return processing. 
Conventional usage of weights is done when the environment is initialized with `continuous_weights=True`. In this case, the returns are
the weighed average of the closing prices, irrespective of the date being a holding date or a rebalancing date (thus modifying the weights the next day).

If `continuous_weights=False`, however, we can split the next weights (*agent action*) in three vectors.

* W buy are the weight increases (positive weight deltas) between consecutive rebalancings. 
* W sell are the weight decreases (negative weight deltas) between consecutive rebalancings.
* W hold is the weight of the instrument that is not sold nor bought. 

The implementation in detail can be found in `rl_factory/common.py decompose_weights_tensor`.
Then, the returns for the frame between the effective action date and the rebalancing date, are

For all the dates except the effective action date, the returns are computed identically as in the `continous_weights=True` case. 
For the effective action date, the return is the sum of three different return sources:

* Buys: the return is computed as the ratio between the Close price and the Open price.
* Sells: the return is computed as the ratio between the Open price, when it is sold and the last Close price.
* Hold: the return is computed as the ratio of the Close price and the last Close.


## Agents

Gymfolio is compatible with most of the Stable Baselines 3 agents, having been tested with a subset of them both with discrete and continuous action spaces. 
Gymfolio also has been successfully used in training decision transformers, generating trajectories to train the agent offline. 

Examples of both scenarios are provided in the `examples/` folder.