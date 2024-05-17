# GymFolio
Welcome to GymFolio, a specialized Reinforcement Learning environment designed for the dynamic field of portfolio optimization. 
GymFolio allows practitioners to create investment strategies for financial investment using Reinforcement Learning. 

Built on Python, this environment provides a framework for training and evaluating various deep reinforcement 
learning models like PPO, SAC, and DQN in the context of managing and optimizing financial portfolios. It is compatible
with most of the agents in the package [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/) and has
been tested with `DQN`, `DDPG`, `PPO` and `SAC`. 

GymFolio offers a highly flexible environment that allows users to experiment with different reward mechanisms, as well as extend it to their own. 
Users can utilize traditional log returns, or incorporate other usual metrics from the industry such as the Sharpe ratio and tracking error to measure performance. 
GymFolio also integrates key financial concepts such as the risk-free rate, slippage, and transaction costs, which are crucial for creating realistic 
trading simulations and optimizing portfolio decisions avoiding usual pitfalls that invalidate promising strategies.



## Base environment
The base environment class is located in `envs/base/PortfolioOptimizationEnv`, and it is built using OHLC data and an additional dataframe for 
the state observations provided by the environment, such as, but not limited to prices, technical indicators or alternative timestamped data.

### Parameters
 * df_ohlc: `pandas Dataframe` with the Open, Close, High and Low prices at the desired granularity.
 * df_observations: `pandas Dataframe` with all the information used as observation of the environment. 
 * rebalance_every: `int, default = 1` time between actions in consecutive time units in the series. `rebalance_every = 4`, for instance, would mean that every 4 available dates a new rabalancing decision will be made.
A value of 1 is for continuous (daily in the default time units) rebalancing.
 * slippage: `float, default = 0.005`, price changes from the submission of the order to its fulfillment.
 * transaction_costs: `float, default = 0.002`, costs of placing an order.
 * continuous_weights: `bool, default = False`, should we consider the weights as continuous, based only on Close to Close price timeseries
   or split them in *Holding weights*, *Buying weights* and *Selling weighs* (More details in a follow-up section)
 * allow_short_positions: `bool, default = False`, enforce long-only portfolios (`False`) or allow short positions.
 * max_trajectory_len: `int, default = 252`, maximum time units for the trajectory.
 * observation_frame_lookback: `int` number of observations to return from the environment to take the next action.
 * render_mode: `str, default=tile` how to return the state of the environment, either as a `vector`(1D), `tile` (2D) or `tensor` (3+D).
 * agent_type: `str` either discrete or continuous.
 * convert_to_terminated_truncated: `bool` use done (old Gym version) or truncated and terminated (new Gymnasium version compatible).

### Environment behavior
Let's assume that we have `rebalance_every=5`, `return_observation_frame=True` with continuous weights and a trajectory of 20.
The environment starts on `2023-03-06` and our first rebalancing date is `2023-03-06 +5 trading days = 2023-03-13`, the next on the `2023-03-20`.

1. An observation frame with dates `2023-03-07`:`2023-03-13` is sliced from df_observations and return to the agent
   in the previous `env.step()`.
2. The agent process the observations and submits an action, the new portfolio weights.
3. As the environment has continuous weights, no further processing is done to them.
4. Return and observation frames are extracted between the next available date from the rebalancing date (`2023-03-14` for `2023-03-13`).
   The observation will be returned by the current `env.step()`.
5. The reward is computed as the weighed average return of each instrument for the time window between the effective date of the action (`2023-03-14`)
   and the next rebalancing date (`2023-03-20`).
6. Date trackers are updated and the environment checks if the trajectory has the maximum length of 10, or if there are no more available rebalancing dates in the history.
   If that is the case, `env.step()` returns `done=True`, else the environment proceeds to the next interaction.


### Base reward
The reward provided is the log return between the effective rebalancing period and the previous rebalancing period.
Some additional considerations on how the return series is computed when `continous_weights=False` are explained
in section [Weight and return processing](###Weight and return processing.)

$ r_t = log(1 + P_t / P_{t-1})$ where $P_t$ is the price of an asset at time t.

Then, as the agent decides through an episode, it collects subsequent $r_t$ rewards at each rebalancing date, with a discount factor $\gamma$.

More sophisticated reward functions are in the children environments located in `envs/custom`, which have the *Sharpe Ratio* (`SharpeEnv`),
*Sortino Ratio* (`SortinoEnv`), *Calmar Ratio* (`CalmarEnv`) and *Tracking error* (`TrackingErrorEnv`).



### Weight and return processing. 
Conventional usage of weights is done when the environment is initialized with `continuous_weights=True`. In this case, the returns are
the weighed average of the returns from the closing price of the previous rebalancing date, to the closing price of the current rebalancing date, 
irrespective of the date being a holding date or a rebalancing date (thus modifying the weights the next day).

$$r=\vec{w}.\cdot \vec{\frac{Price_{Close,t+1}}{Price_{Close,t}}}$$

If `continuous_weights=False`, however, we can split the next weights (*agent action*) in three vectors.

![weights](./docs/_static/weights.png)

* W buy are the weight increases (positive weight deltas) between consecutive rebalancings. 
* W sell are the weight decreases (negative weight deltas) between consecutive rebalancings.
* W hold is the weight of the instrument that is not sold nor bought. It is the minimum value between the old and new weights for each instrument.

The implementation in detail can be found in `src/envs/common.py decompose_weights_tensor`.

Then, the returns for the frame between the effective action date and the rebalancing date, are computed 

For all the dates except the effective action date, the returns are computed identically as in the `continous_weights=True` case. 
For the effective action date, the return is the sum of three different return sources:

* Buys: the return is computed as the ratio between the Close price and the Open price.
* Sells: the return is computed as the ratio between the Open price, when it is sold and the last Close price.
* Hold: the return is computed as the ratio of the Close price and the last Close.

![returns](./docs/_static/returns.png)


### Other considerations
- **Risk-free rate**: The return on an investment with zero risk, providing a baseline against which to measure other investments.
It is used in variants like the Sharpe ratio environment. 

- **Slippage**: The difference between the expected price of a trade and the price at which the trade is executed.
It is used in the `continous_weights=True` case as a negative contribution in the returns of the first day. In the `continous_weights=False` it 
only impacts the buys and sales. 

- **Transaction costs**: Expenses incurred when buying or selling securities, including commissions and fees.
  As with the slippage, t is used in the `continous_weights=True` case as a negative contribution in the returns of the first day. In the `continous_weights=False` it
  only impacts the buys and sales.

## Other environments

### Sharpe ratio environment 
The Sharpe ratio environment replaces the cumulative log-returns as the reward by the Sharpe ratio between periods. Code is located in 
`src/envs/custom/SharpeEnv`. 

The Sharpe ratio is computed as:

$$S = (R_p - R_f) / \sigma_p$$ 

where $R_p$ is the expected annualized portfolio return, $R_f$ is the risk-free rate, and $\sigma_p$ is the standard deviation of the portfolio's excess return.

This environment only modifies the `compute_reward` method of the environment class and adds three additional attributes:

* `riskfree_rate`: risk free rate to compute the numerator of the sharpe ratio (r-Rf).
* `periods_per_year`: periods per year to annualize returns for the Sharpe ratio computation.
* `compute_cumulative`: use the whole trajectory of the episode or just the last returns for the computation of the Sharpe ratio.


### Tracking error environment 
The tracking error environment replaces the reward by the tracking error between the portfolio returns and a series of reference returns, usually an index like the 
SP500. Code is located in `src/envs/custom/TrackingErrorEnv`.

The tracking error is computed as the standard deviation between the portfolio returns and the reference returns.
This environment modifies the `compute_reward` and `step` methods of the environment class and adds one additional attribute:

* `df_reference`: pd.DataFrame with the reference returns (tracked instrument).



## Agents

The goal of Gymfolio is to provide the main building blocks and wrappers for portfolio optimization, 
that can be used and extended with low or high level implementations of RL agents  using any python library (numpy, Tensorflow, PyTorch, PTAN or StableBaselines3).

Gymfolio is compatible with most of the Stable Baselines 3 agents, and has been tested with a subset of them both with discrete and continuous action spaces. 
Gymfolio also has been successfully used in training Decision Transformers, generating trajectories to train the agent offline.  

## Example
An example of the usage of StableBaselines3 is provided in `examples/stable_baselines3.py`. It consumes the `data/example.h5` datasets in HDF5 format. 