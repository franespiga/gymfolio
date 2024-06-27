"""
Custom Portfolio optimization environments
"""
import pandas as pd
import numpy as np
import torch
from .base import PortfolioOptimizationEnv
from .metrics import sharpe_ratio, tracking_error
from .common import create_return_matrices, decompose_weights_tensor

class SharpeEnv(PortfolioOptimizationEnv):
    """
    This class implements a custom environment following the `gym` structure for Portfolio Optimization.
    Using the Sharpe ratio as the rewards

    :param pd.DataFrame df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
    :param pd.DataFrame df_observations: pd.DataFrame with the environment features.
    :param int rebalance_every: periods between consecutive rebalancing actions.
    :param float slippage: %loss due to gap between decision price for the agent and the execution price.
    :param float transaction_costs: %loss due to execution of the trade.
    :param bool continuous_weights: `True` to split the weights in (h)eld, (b)ought and (s)old positions.
    :param bool allow_short_positions: `True` to enable short positions.
    :param int max_trajectory_len: max total number of periods for the trajectories. E.g. 252 for a trading year.
    :param int observation_frame_lookback: return the previous N observations from the environment to take the next action.
    :param str render_mode: Either `tile` (2D), `tensor`(+2D) or `vector` (1D) to return the environment state.
    :param str agent_type: `discrete` or `continuous`.
    :param bool convert_to_terminated_truncated: use done (old Gym version) or truncated and terminated (new Gymnasium version)
    :param float riskfree_rate: risk free rate to compute the numerator of the sharpe ratio (r-Rf)
    :param int periods_per_year: periods per year to annualize returns for the Sharpe ratio computation.
    :param bool compute_cumulative: use the whole trajectory of the episode or just the last returns for the computation
    :param verbose: verbosity (0: None, 1: error messages, 2: all messages)
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
                 render_mode: str = 'tile',
                 agent_type: str = 'discrete',
                 convert_to_terminated_truncated: bool = False,
                 riskfree_rate: float = 0.0,
                 periods_per_year:int = 252,
                 compute_cumulative:bool = False,
                 verbose: int = 0,
                 ):

        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)

        self.riskfree_rate = riskfree_rate
        self.periods_per_year = periods_per_year
        self.compute_cumulative = compute_cumulative

    def compute_reward(self, r) -> torch.Tensor:
        """
        :param r: returns series
        :return: sharpe ratio of the returns or the full trajectory
        """
        return sharpe_ratio(r if not self.compute_cumulative else self.trajectory_returns,
                            self.riskfree_rate, self.periods_per_year)



class TrackingErrorEnv(PortfolioOptimizationEnv):
    """
     This class implements a custom environment following the `gym` structure for Portfolio Optimization.
     Using the Tracking error as the reward

    :param pd.DataFrame df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
    :param pd.DataFrame df_observations: pd.DataFrame with the environment features.
    :param pd.DataFrame df_reference: pd.DataFrame with the reference returns (tracked instrument).
    :param int rebalance_every: periods between consecutive rebalancing actions.
    :param float slippage: %loss due to gap between decision price for the agent and the execution price.
    :param float transaction_costs: %loss due to execution of the trade.
    :param bool continuous_weights: `True` to split the weights in (h)eld, (b)ought and (s)old positions.
    :param bool allow_short_positions: `True` to enable short positions.
    :param int max_trajectory_len: max total number of periods for the trajectories. E.g. 252 for a trading year.
    :param int observation_frame_lookback: return the previous N observations from the environment to take the next action.
    :param str render_mode: Either `tile` (2D), `tensor`(+2D) or `vector` (1D) to return the environment state.
    :param str agent_type: `discrete` or `continuous`.
    :param bool convert_to_terminated_truncated: use done (old Gym version) or truncated and terminated (new Gymnasium version)
    :param verbose: verbosity (0: None, 1: error messages, 2: all messages)
     """

    def __init__(self,
                 df_ohlc: pd.DataFrame,
                 df_observations: pd.DataFrame,
                 df_reference: pd.DataFrame,
                 rebalance_every: int = 1,
                 slippage: float = 0.005,
                 transaction_costs: float = 0.002,
                 continuous_weights: bool = False,
                 allow_short_positions: bool = False,
                 max_trajectory_len: int = 252,
                 observation_frame_lookback: int = 5,
                 render_mode: str = 'tile',
                 agent_type: str = 'discrete',
                 convert_to_terminated_truncated: bool = False,
                 verbose: int = 0
                ):

        self.df_reference = df_reference
        self.reference_returns = None
        self.trajectory_reference_returns = None
        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)

    def reset(self, seed: int = 5106, options: dict = None) -> tuple:
        """
        Method to reset the environment, it empties the trajectory reference returns.
        :param int seed: random seed.
        :param dict options: dictionary of options
        :return:
        """
        self.trajectory_reference_returns = []
        obs, info = super().reset()
        return obs, info

    def compute_reward(self, r) -> torch.Tensor:
        """
        Computes the tracking error. It clips the values for stability and helping convergence.
        :param pd.Series r: returns series
        :return: tracking error of the episode trajectory VS the reference trajectory.
        """
        if isinstance(self.trajectory_returns, list):
            trajectory_rets = pd.concat(self.trajectory_returns)
        else:
            trajectory_rets = self.trajectory_returns

        trajectory_rets = torch.Tensor(trajectory_rets).squeeze()

        reference_rets = pd.concat(self.trajectory_reference_returns).drop_duplicates()
        reference_rets = reference_rets.tail(reference_rets.shape[0]-1)
        reference_rets = torch.Tensor(reference_rets).squeeze()

        terror = tracking_error(trajectory_rets, reference_rets)
        creward = torch.clip(torch.pow(terror, -1), 0, 1e5)
        return creward

    def step(self, action) -> tuple:
        """
        Environment step method. Takes the agent's action and returns the new state, reward and whether or not the episode has finished.
        This method allows legacy behavior of gym when an episode has finished, returning the `done` boolean flag, and the new
        gymnasium convention of separating the `done` flag into `terminated` (the episode reached the max length allowed) or
        `truncated` if it ended for different reasons.

        The step method does the following sequence of operations in the base environment

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

        Then, in the outer part of the `step()` method:

        1. The return tuple from the base environment is unrolled.

        2. The tracking error reward is computed.

        3. New boolean flags for done  / terminated and truncated are computed.

        :param int or float action: index of the action for `discrete` agents or float values for each dimension of the action space for continuous agents.
        :return: the new state, reward, done flag (or truncated and terminated) and information.
        :rtype: tuple
        """
        self.reference_returns = self.df_reference[self.current_rebalancing_date:self.next_rebalancing_date]
        self.trajectory_reference_returns.append(self.reference_returns)

        _ = super().step(action)

        if not self.convert_to_terminated_truncated:
            observations, reward, done, info = _
        else:
            observations, reward, truncated, terminated, info = _

        reward = self.compute_reward(self.reference_returns)

        if self.convert_to_terminated_truncated:
            return observations, reward, truncated, terminated, info
        else:
            return observations, reward, done, info



def reward_criterion_scorer(func, r_a, r_b, threshold, **kwargs):
    """
    This function computes the normalized score of a reward criterion to compare 2 sets of portfolio returns
    :param func: scoring function
    :param r_a: returns of the first strategy
    :param r_b: returns of the second strategy
    :return:
    """
    sc_a = func(r_a, **kwargs)
    sc_b = func(r_b, **kwargs)

    ratio = sc_a/sc_b

    if np.abs(ratio-1) <= threshold:
        return 0.5
    elif ratio-1 > 0:
        return 1
    else:
        return 0

class CompositeRewardEnv(PortfolioOptimizationEnv):
    """
     This class implements a custom environment following the `gym` structure for Portfolio Optimization.
     Using a composite reward mechanism and pseudo-inverse rl approach.

     The composite reward is a set of functions that take the returns of two different strategies and provide a score
     across the individual criteria and aggregate them to determine if there is an overall winner (reward=0 or 1)
     or if both are comparable (0.5)

    :param pd.DataFrame df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
    :param pd.DataFrame df_observations: pd.DataFrame with the environment features.
    :param list composite_reward_funcs: list of criteria functions to compute the composite reward with.
    :param float composite_reward_threshold: threshold to consider comparable results (0.5 +/- threshold).
    :param int rebalance_every: periods between consecutive rebalancing actions.
    :param float slippage: %loss due to gap between decision price for the agent and the execution price.
    :param float transaction_costs: %loss due to execution of the trade.
    :param bool continuous_weights: `True` to split the weights in (h)eld, (b)ought and (s)old positions.
    :param bool allow_short_positions: `True` to enable short positions.
    :param int max_trajectory_len: max total number of periods for the trajectories. E.g. 252 for a trading year.
    :param int observation_frame_lookback: return the previous N observations from the environment to take the next action.
    :param str render_mode: Either `tile` (2D), `tensor`(+2D) or `vector` (1D) to return the environment state.
    :param str agent_type: `discrete` or `continuous`
    :param bool convert_to_terminated_truncated: use done (old Gym version) or truncated and terminated (new Gymnasium version)
    :param verbose: verbosity (0: None, 1: error messages, 2: all messages)
     """

    def __init__(self,
                 df_ohlc: pd.DataFrame,
                 df_observations: pd.DataFrame,
                 composite_reward_funcs: list,
                 composite_reward_threshold: float = 0.1,
                 rebalance_every: int = 1,
                 slippage: float = 0.005,
                 transaction_costs: float = 0.002,
                 continuous_weights: bool = False,
                 allow_short_positions: bool = False,
                 max_trajectory_len: int = 252,
                 observation_frame_lookback: int = 5,
                 render_mode: str = 'tile',
                 agent_type: str = 'discrete',
                 convert_to_terminated_truncated: bool = False,
                 verbose: int = 0,
                 **kwargs
                 ):

        self.kwargs = kwargs
        self.prior_weights = None
        self.composite_reward_funcs = composite_reward_funcs
        self.composite_reward_threshold = composite_reward_threshold
        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)

    def reset(self, seed: int = 5106, options: dict = None) -> tuple:
        """
        Method to reset the environment, it empties the trajectory reference returns.
        :param int seed: random seed.
        :param dict options: dictionary of options
        :return:
        """
        obs, info = super().reset()
        return obs, info

    def compute_composite_rewards(self, r_new, r_old) -> torch.Tensor:
        """
        Computes the composite reward.
        Each function takes the returns with the new and former portfolio weights
        :param pd.Series r: returns series
        :return: tracking error of the episode trajectory VS the reference trajectory.
        """
        c = 0
        for func in self.composite_reward_funcs:
            score = reward_criterion_scorer(func,  r_new, r_old,
                                            self.composite_reward_threshold, **self.kwargs)
            c += score

        c /= len(self.composite_reward_funcs)

        if c >= 0.5+self.composite_reward_threshold:
            return 1
        elif c < 0.5-self.composite_reward_threshold:
            return 0
        else:
            return 0.5


    def step(self, action) -> tuple:
        """
        Environment step method. Takes the agent's action and returns the new state, reward and whether or not the episode has finished.
        This method allows legacy behavior of gym when an episode has finished, returning the `done` boolean flag, and the new
        gymnasium convention of separating the `done` flag into `terminated` (the episode reached the max length allowed) or
        `truncated` if it ended for different reasons.

        The step method does the following sequence of operations in the base environment

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

        Then, in the outer part of the `step()` method:

        1. The return tuple from the base environment is unrolled.

        2. The tracking error reward is computed.

        3. New boolean flags for done  / terminated and truncated are computed.

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
        effective_rebalancing_date = self.available_dates[self.available_dates.index(self.current_rebalancing_date) + 1]
        r_sell = torch.Tensor(self.returns_sell.loc[[effective_rebalancing_date]].values).squeeze()
        r_buy = torch.Tensor(self.returns_buy.loc[[effective_rebalancing_date]].values).squeeze()
        return_frame = self.returns_hold.loc[effective_rebalancing_date:self.next_rebalancing_date, :]
        R_hold = torch.Tensor(return_frame.values)
        r_hold = torch.Tensor(self.returns_hold.loc[[effective_rebalancing_date]].values).squeeze()

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
            self.next_rebalancing_date = self.rebalancing_dates[
                self.rebalancing_dates.index(self.current_rebalancing_date) + 1]  # Get next rebalancing date

        # Compute split weights and compute returns ---
        if self.continuous_weights:
            r = torch.matmul(self.new_weights, R_hold.T)
            r[0] += - (self.transaction_costs + self.slippage)

            r_o = torch.matmul(self.current_weights, R_hold.T)
            r_o[0] += - (self.transaction_costs + self.slippage)
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

            # Compute returns with previous weights
            r_o = torch.matmul(self.current_weights, R_hold.T)

        self.last_returns = r
        self.prior_returns = r_o

        df_r = pd.Series(r.numpy().squeeze(), index=pd.to_datetime(return_frame.index))
        self.trajectory_returns.append(df_r)
        if truncated or terminated:
            self.trajectory_returns = pd.concat(self.trajectory_returns)

        reward = self.compute_composite_rewards(r, r_o)

        if self.convert_to_terminated_truncated:
            return observations, reward, truncated, terminated, info
        else:
            return observations, reward, truncated or terminated, info