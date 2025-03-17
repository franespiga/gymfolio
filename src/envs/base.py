"""
Gymfolio. A Reinforcement Learning environment for Portfolio Optimization
Copyright (C) 2024 Francisco Espiga Fernández

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pandas as pd
import random
import torch
from .common import create_return_matrices, decompose_weights_tensor
import gymnasium
from gymnasium.spaces import Box, Discrete
import collections


class PortfolioOptimizationEnv(gymnasium.Env):
    """
    This class implements a custom environment following the `gym` structure for Portfolio Optimization.
    The vanilla environment uses the average return as the reward
    """

    def __init__(self,
                 df_ohlc: pd.DataFrame,
                 df_observations: pd.DataFrame,
                 rebalance_every: int = 1,
                 slippage: float = 0.0005,
                 transaction_costs: float = 0.0002,
                 continuous_weights: bool = False,
                 allow_short_positions: bool = False,
                 max_trajectory_len: int = 252,
                 observation_frame_lookback: int = 5,
                 action_delay:int = 1,
                 render_mode: str = 'tile',
                 agent_type: str = 'discrete',
                 convert_to_terminated_truncated: bool = False,
                 trajectory_bootstrapping: bool = False,
                 episodic_instrument_shifting: bool = False,
                 verbose: int = 0,
                 ):
        """
        :param pd.DataFrame df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
        :param pd.DataFrame df_observations: pd.DataFrame with the environment features.
        :param int rebalance_every: periods between consecutive rebalancing actions.
        :param float slippage: %loss due to gap between decision price for the agent and the execution price.
        :param float transaction_costs: %loss due to execution of the trade.
        :param bool continuous_weights: `False` to split the weights in (h)eld, (b)ought and (s)old positions.
        :param bool allow_short_positions: `True` to enable short positions.
        :param int max_trajectory_len: max total number of periods for the trajectories. E.g. 252 for a trading year.
        :param int observation_frame_lookback: return the previous N observations from the environment to take the next action.
        :param int action_delay: number of periods from decision to execution. Defaults to 1 (e.g. next trading day)
        :param str render_mode: Either `tile` (2D), `tensor`(3D) or `vector`(1D) to return the environment state.
        :param str agent_type: `discrete` or `continuous`
        :param bool convert_to_terminated_truncated: use done (old Gym version) or truncated and terminated (new Gymnasium version)
        :param bool trajectory_bootstrapping: use non-consecutive ordered rebalancing dates to break correlation (trajectory bootstrapping)
        :param bool episodic_instrument_shifting: shift order of instruments at env.reset()
        :param verbose: verbosity (0: None, 1: error messages, 2: all messages)
        """
        self.indicator_instrument_names = None
        self.indicator_names = None
        self.global_columns = None
        self.indicator_columns = None
        self.returns_sell = None
        self.returns_buy = None
        self.returns_hold = None
        self.df_ohlc = df_ohlc.dropna()
        self.n_instruments = self.df_ohlc.loc[:, [i for i in self.df_ohlc.columns if i[1] == 'Close']].shape[1]
        self.episodic_instrument_shifting = episodic_instrument_shifting

        self.df_observations = df_observations
        self.rebalance_every = rebalance_every
        self.slippage = slippage
        self.transaction_costs = transaction_costs
        self.continuous_weights = continuous_weights
        self.allow_short_positions = allow_short_positions
        # Date management ---------------------------------
        self.action_delay = action_delay
        self.available_dates = self.df_ohlc.sort_index().index.values.tolist()
        self.rebalancing_dates = None
        self.delayed_action_dates = None
        self.current_rebalancing_date = None
        self.next_rebalancing_date = None
        # State observation management --------------------
        self.render_mode = render_mode
        self.agent_type = agent_type
        self.convert_to_terminated_truncated = convert_to_terminated_truncated
        self.trajectory_bootstrapping = trajectory_bootstrapping

        self.max_trajectory_len = max_trajectory_len
        self.observation_frame_lookback = observation_frame_lookback
        self.current_trajectory_len = None
        self.trajectory_returns = []
        self.last_returns = None

        self.action_space = self.get_action_space()
        self.action_size = self.action_space.n if isinstance(self.action_space, Discrete) else self.action_space.shape[
            0]
        self.observation_space = self.get_observation_space()

        self.current_weights = np.zeros(self.action_size)
        self.new_weights = np.zeros(self.action_size)

        self.process_indicator_types()
        self.preprocess_returns()
        self.reset()

        self.verbose = verbose

    """
    Environment auxiliary methods
    """

    def get_action_space(self):
        """
        Infer action space as the number of instruments to choose from. This method is used by StableBaselines3 to
        setup the network architecture of the agents.
        Instruments should be provided to the environment using a pd.DataFrame with MultiIndex columns [(Ticker, Open), (Ticker,...),(Ticker, Close)]

        :return: number of "actions" (instruments) and their bounds in the continuous action space case.
        """
        n_instruments = len(set([i[0] for i in self.df_ohlc.columns]))
        if self.agent_type == 'discrete':
            return Discrete(n_instruments)
        elif self.agent_type == 'continuous':
            return Box(-np.ones(n_instruments) * self.allow_short_positions, np.ones(n_instruments))

    def get_observation_space(self):
        """
        Get the observation space for the agents. This method is used by StableBaselines3 to
        setup the network architecture of the agents.
        :return: observation space with the bounds across each element of the state.

        """
        if self.render_mode == 'vector':
            lows = np.tile(self.df_observations.min(axis=0).values, (1 + self.observation_frame_lookback, 1)).squeeze()
            highs = np.tile(self.df_observations.max(axis=0).values, (1 + self.observation_frame_lookback, 1)).squeeze()
            return Box(low=lows, high=highs, shape=[1, (1 + self.observation_frame_lookback)*self.df_observations.shape[1]])
        if self.render_mode == 'tile':
            lows = np.tile(self.df_observations.min(axis=0).values, (1 + self.observation_frame_lookback, 1))
            highs = np.tile(self.df_observations.max(axis=0).values, (1 + self.observation_frame_lookback, 1))
            return Box(low=lows, high=highs, shape=[1 + self.observation_frame_lookback, self.df_observations.shape[1]])
        elif self.render_mode == 'tensor':
            new_shape = (len(self.indicator_names), len(self.indicator_instrument_names))
            lows = np.tile(np.reshape(self.df_observations.min(axis=0).values, new_shape),
                           (1 + self.observation_frame_lookback, 1, 1))
            lows = lows.transpose(2, 0, 1)
            highs = np.tile(np.reshape(self.df_observations.max(axis=0).values, new_shape),
                            (1 + self.observation_frame_lookback, 1, 1))
            highs = highs.transpose(2, 0, 1)
            return Box(low=lows, high=highs,
                       shape=[len(self.indicator_instrument_names), 1 + self.observation_frame_lookback,
                              len(self.indicator_names)])
        else:
            raise ValueError(f"render_mode={self.render_mode} not supported. Should be tile/tensor/vector")

    def preprocess_returns(self) -> None:
        """
        Transform OHLC prices into buy/hold/sell returns
        - Buy: Close_t/Open_t
        - Hold: Close_t/Close_t-1
        - Sell: Open_t/Close_t-1

        It transforms the df_ohlc attribute and stores in three dataframes the Hold/Buy/Sell returns per instrument.

        :return: None
        """
        ohlc_cols = list(set([i[1] for i in self.df_ohlc.columns]))
        instrument_ordering = [(i, j) for i in self.indicator_instrument_names for j in ohlc_cols]
        r_h, r_b, r_s = create_return_matrices(self.df_ohlc.loc[:,instrument_ordering])
        self.returns_hold = r_h
        self.returns_buy = r_b
        self.returns_sell = r_s

    def process_indicator_types(self) -> None:
        """
        This auxiliary method processes the df_observations dataframe to extract per-instrument indicators and global indicators.
        This method is used to ensure that tensor construction for convolutional feature extractors as per the SB3 conventions
        is correct

        :return: None
        """
        indicator_names = [i[1] for i in self.df_observations.columns]
        occurrences = collections.Counter(indicator_names)
        single = [k for k, v in occurrences.items() if v == 1]
        multiple = [k for k, v in occurrences.items() if v > 1]

        indicator_columns = [i for i in self.df_observations.columns if i[1] in multiple]
        self.global_columns = [i for i in self.df_observations.columns if i[1] in single]

        self.indicator_names = list(set([i[1] for i in indicator_columns]))
        self.indicator_instrument_names = list(set([i[0] for i in indicator_columns]))

        if self.episodic_instrument_shifting:
            random.shuffle(self.indicator_instrument_names)

        # Reorder indicators as [(Instrument A, indicator 1), (Instrument A, indicator 2) ... (Instrument M, indicator N)]
        # To be used with Convolutional feature extractors in the policy
        self.indicator_columns = [(i, j) for i in self.indicator_instrument_names for j in self.indicator_names if (i,j) in self.df_observations.columns]
        self.df_observations = self.df_observations.loc[:,self.indicator_columns]


    def compute_reward(self, r) -> torch.Tensor:
        """
        Compute the rewards as the sum of log-returns.
        :param torch.Tensor r: returns series
        :return: sum of log returns.
        :rtype: torch.Tensor of type float.
        """
        return torch.sum(torch.log(1 + r))

    def create_info(self):
        """
        Placeholder method to return additional environment info for custom environments
        :return: info dictionary with metadata.
        :rtype: dict
        """
        return dict()

    def reset(self, seed: int = 5106, options: dict = None) -> tuple:
        """
        Method to reset the environment, as per the gym(nasium) conventions. It takes a random date in the history
        and prepares the rebalancing dates based on the `rebalance_every` attribute.

        Weights are initialized randomly at the beginning of every episode, which has been proved to speed up training,
        compared to an initial 100% allocation to a single instrument (e.g. cash) or zero.


        :param seed: int, random seed.
        :param options: dictionary of options
        :return: initial observation for the episode and dictionary with information.
        :rtype: tuple
        """
        self.current_rebalancing_date = random.choice(self.available_dates[:-(self.rebalance_every + 1)])
        delayed_idx = self.available_dates.index(self.current_rebalancing_date)+self.action_delay
        self.current_delayed_action_date = self.available_dates[delayed_idx]

        self.current_trajectory_len = 0.0
        self.trajectory_returns = []

        # This is required for episodic instrument shifting
        self.process_indicator_types()
        self.preprocess_returns()

        if self.trajectory_bootstrapping:
            n_samples = 1+self.max_trajectory_len//self.rebalance_every
            self.rebalancing_dates = np.random.choice(
                self.available_dates[self.available_dates.index(self.current_rebalancing_date):],
                n_samples)
            self.rebalancing_dates.sort()
        else:
            self.rebalancing_dates = self.available_dates[
                                     self.available_dates.index(self.current_rebalancing_date)::self.rebalance_every]

            self.delayed_action_dates = self.available_dates[
                         self.available_dates.index(self.current_delayed_action_date)::self.rebalance_every]

        self.next_rebalancing_date = self.rebalancing_dates[self.rebalancing_dates.index(self.current_rebalancing_date) + 1]
        self.next_delayed_action_date = self.delayed_action_dates[self.delayed_action_dates.index(self.current_delayed_action_date) + 1]

        _new_weights = torch.rand(self.action_size)
        self.new_weights = _new_weights / _new_weights.sum()  # We start without any assets

        idx_lookback = max(0,
                           self.available_dates.index(self.current_rebalancing_date) - self.observation_frame_lookback)

        observation_frame = self.df_observations[self.available_dates[idx_lookback]:self.current_rebalancing_date]
        observation_frame = self.expand_observation_frame(observation_frame)
        # Add any additional information ---
        info = dict()
        info['indices'] = observation_frame.index.tolist()
        info['features'] = observation_frame.columns.tolist()

        if self.render_mode == 'tile':
            observation_frame = self.return_obs_frame_as_tile(observation_frame)
        elif self.render_mode == 'vector':
            observation_frame = self.return_obs_frame_as_vector(observation_frame)
        elif self.render_mode == 'tensor':
            observation_frame = self.return_obs_frame_as_tensor(observation_frame)

        return observation_frame, info

    def expand_observation_frame(self, obs_frame):
        """
        Method to control the observations at points in time where `observation_frame_lookback+1` observations are not
        available. When there are no available observations, the method returns the last observation of the environment (df_observations)
        repeated `observation_frame_lookback+1` times.
        When the available observations are lower than `observation_frame_lookback+1`, it returns a random sample of the expected size.
        In any other case, the original observation frame is returned.

        :param pd.DataFrame obs_frame: observation frame
        :return: an observation frame with the expected `observation_frame_lookback+1` number of observations
        :rtype: pd.DataFrame
        """

        if obs_frame.shape[0] == 0:
            return self.df_observations.tail(1).sample(self.observation_frame_lookback + 1, replace=True)
        elif obs_frame.shape[0] < self.observation_frame_lookback + 1:
            return obs_frame.sample(self.observation_frame_lookback + 1, replace=True)
        else:
            return obs_frame

    def return_obs_frame_as_tensor(self, obs_frame) -> torch.Tensor:
        """
        This method takes the observation frame and returns it as a 3+D tensor, to be used for example with the CNN
        feature extractors of stable baselines3. It relies on the auxiliary method `process_indicator_types` to identify
        the global indicators and instrument specific indicators to shape the tensor. Global indicators (information available across instruments)
        are repeated `n_instruments` times and concatenated with the indicators of each instrument.

        :param pd.DataFrame obs_frame:
        :return: tensor with the merged observations.
        :rtype: torch.Tensor
        """
        # Take repeated indicators and reshape that
        # Tensors are in the shape of Channels x Height x Width -> instruments x lookback x indicators
        n_channels = len(self.indicator_instrument_names)
        global_tensor = torch.tile(torch.Tensor(obs_frame.loc[:, self.global_columns].values), [n_channels, 1, 1])

        indicators_tensor = torch.Tensor(
            [obs_frame[i].loc[:, self.indicator_names].values for i in self.indicator_instrument_names])

        if not 0 in global_tensor.size():
            return torch.concat([indicators_tensor, global_tensor], 1)
        else:
            return indicators_tensor

    def return_obs_frame_as_vector(self, obs_frame) -> torch.Tensor:
        """
        This method returns the observation frame as a 1D vector.
        :param pd.DataFrame obs_frame: environment observation frame with `lookback_window+1` rows.
        :return: 1D vector of the environment state.
        :rtype: torch.Tensor
        """
        obs = torch.Tensor(obs_frame.values.squeeze())
        return obs

    def return_obs_frame_as_tile(self, obs_frame) -> torch.Tensor:
        """
        This method returns the observation frame as a 2D vector.
        :param pd.DataFrame obs_frame: environment observation frame with `lookback_window+1` rows and `indicators` columns.
        :return: 2D matrix of the environment state.
        :rtype: torch.Tensor
        """
        obs = torch.Tensor(obs_frame.values)
        return obs

    def step(self, action) -> tuple:
        """
        Environment step method. Takes the agent's action and returns the new state, reward and whether or not the episode has finished.
        This method allows legacy behavior of gym when an episode has finished, returning the `done` boolean flag, and the new
        gymnasium convention of separating the `done` flag into `terminated` (the episode reached the max length allowed) or
        `truncated` if it ended for different reasons.

        The step method does the following sequence of operations.

        1. Compute the new weights based on the action taken.
        Depending on the agent, it can be either a full allocation to an instrument (`discrete`), 1-sum weights (`continuous`
        action space with long-only positions) or 0-sum weights (`continuous` action space with dollar-neutral positions).

        2. Compute the returns of the held, bought and sold positions.
        - Held positions return are computed as the price at last closing period divided by the initial open period.
        - Bought positions are bought at the Open. Return is the close by the open prices of that period.
        - Sold positions are sold at the Open. Return is the price at the open divided by the previous period close.

        3. Get the observation frame and expand if necessary. Transform into vector (1D), tile (2D) or tensor (3+D).

        4. Check if the episode has ended (truncated / terminated) and update the date tracker.

        5. Split the weights in held, bought and sold positions to compute the return series and append to the returns dataframe.

        6. Return the new state, reward, done flag (or truncated and terminated) and information.

        :param int or float action: index of the action for `discrete` agents or float values for each dimension of the action space for continuous agents.
        :return: the new state, reward, done flag (or truncated and terminated) and information.
        :rtype: tuple
        """
        # Check that weights have the correct dimension ---

        # assert action.shape[1] == self.action_size
        self.current_weights = self.new_weights

        if self.agent_type == 'discrete':
            self.new_weights = torch.zeros(self.action_size)
            self.new_weights[action] = 1.0
        else:
            if not self.allow_short_positions:
                self.new_weights = torch.Tensor(action) / torch.Tensor(action).sum()
            else:
                self.new_weights = torch.Tensor(action) - torch.Tensor(action).mean()

                # Get the observation frame ---
        """
        Observation frame takes from the Action date (consecutive timestamp from previous decision date where buys/sell became effective, 
        to the decision date at close)
        """
        # get trajectory returns during holding period ---
        #<FE>: I need to decouple decision and execution to more than 1 period.

        effective_rebalancing_date = self.available_dates[self.available_dates.index(self.current_rebalancing_date) + 1]
        effective_delayed_decision_date = self.available_dates[self.available_dates.index(self.current_delayed_action_date) + 1]



        # Returns ------------
        r_sell = torch.Tensor(self.returns_sell.loc[[effective_delayed_decision_date]].values).squeeze()
        r_buy = torch.Tensor(self.returns_buy.loc[[effective_delayed_decision_date]].values).squeeze()

        return_frame = self.returns_hold.loc[effective_delayed_decision_date:self.next_delayed_action_date, :]
        R_hold = torch.Tensor(return_frame.values)
        r_hold = torch.Tensor(self.returns_hold.loc[[effective_delayed_decision_date]].values).squeeze()


        # Observation frame is the next information available, from our action date (decision date+1) to the next rebalancing date
        # This is the information that we will use to decide the next weights.
        idx_lookback = max(0, self.available_dates.index(self.next_rebalancing_date) - self.observation_frame_lookback)
        observation_frame = self.df_observations[self.available_dates[idx_lookback]:self.next_rebalancing_date]
        observation_frame = self.expand_observation_frame(observation_frame)

        # Add any additional information ---
        info = dict()
        info['indices'] = observation_frame.index.tolist()
        info['features'] = observation_frame.columns.tolist()

        if self.render_mode == 'tile':
            observations = self.return_obs_frame_as_tile(observation_frame)
        elif self.render_mode == 'vector':
            observations = self.return_obs_frame_as_vector(observation_frame)
        elif self.render_mode == 'tensor':
            observations = self.return_obs_frame_as_tensor(observation_frame)

        # Check if the environment is done ---
        """
        If the next rebalancing date is the last one, we will not be able to compute the rewards afterwards. 
        """
        self.current_trajectory_len += self.rebalance_every
        truncated = False
        terminated = False
        if (self.next_rebalancing_date == self.rebalancing_dates[-1]):
            # We do not rebalance in the last rebalancing date, we just want to keep homogeneous decision tensors.
            terminated = True
        elif (self.current_trajectory_len == self.max_trajectory_len):
            truncated = True
        else:
            # move the date cursor to the next rebalancing dates ---
            self.current_rebalancing_date = self.next_rebalancing_date  # Date when we are taking the decision
            self.current_delayed_action_date = self.next_delayed_action_date # Date when we execute the decision
            self.next_rebalancing_date = self.rebalancing_dates[
                self.rebalancing_dates.index(self.current_rebalancing_date) + 1]  # Get next rebalancing date
            self.next_delayed_action_date = self.delayed_action_dates[
                self.delayed_action_dates.index(self.current_delayed_action_date) + 1]  # Get next rebalancing date

        # Compute split weights and compute returns ---
        if self.continuous_weights:
            r = torch.matmul(self.new_weights, R_hold.T)
            r[0] += - (self.transaction_costs + self.slippage)

        else:
            w_h, w_b, w_s = decompose_weights_tensor(self.new_weights, self.current_weights)
            # Compute returns of Buy/Sell
            r_s = torch.dot(w_s.squeeze(), r_sell)
            r_b = torch.dot(w_b.squeeze(), r_buy)
            r_h = torch.dot(w_h.squeeze(), r_hold)

            # Multiply returns x weights to get the portfolio returns
            r = torch.matmul(self.new_weights, R_hold.T)

            # Modify the returns of the 1st day of trajectory to account for slippage and transaction costs
            # as well as buys/sales returns
            r[0] += (r_s - self.transaction_costs - self.slippage)
            r[0] += (r_b - self.transaction_costs - self.slippage)
            r[0] += r_h

        self.last_returns = r

        df_r = pd.Series(r.numpy().squeeze(), index=pd.to_datetime(return_frame.index))
        self.trajectory_returns.append(df_r)
        if truncated or terminated:
            self.trajectory_returns = pd.concat(self.trajectory_returns)
        reward = self.compute_reward(r)

        if self.convert_to_terminated_truncated:
            return observations, reward, truncated, terminated, info
        else:
            return observations, reward, truncated or terminated, info
