"""
Custom Portfolio optimization environments
"""
import numpy as np
import pandas as pd
import random
import torch
from .base import PortfolioOptimizationEnv
from .metrics import sharpe_ratio, sortino_ratio, calmar_ratio, tracking_error


class SharpeEnv(PortfolioOptimizationEnv):
    """
    This class implements a custom environment following the `gym` structure for Portfolio Optimization.
    Using the Sharpe ratio as the rewards
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

    def compute_reward(self, r):
        """
        :param r: returns series
        :return: sharpe ratio of the returns or the full trajectory
        """
        return sharpe_ratio(r if not self.compute_cumulative else self.trajectory_returns,
                            self.riskfree_rate, self.periods_per_year)


class SortinoEnv(PortfolioOptimizationEnv):
    """
    This class implements a custom environment following the `gym` structure for Portfolio Optimization.
    Using the Sortino ratio as the rewards
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
                 periods_per_year: int = 252,
                 compute_cumulative: bool = False,
                 verbose: int = 0,
                 ):

        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)


        self.riskfree_rate = riskfree_rate
        self.periods_per_year = periods_per_year
        self.compute_cumulative = compute_cumulative

    def compute_reward(self, r):
        """
        :param r: returns series
        :return: sortino ratio of the returns or the full trajectory
        """
        return sortino_ratio(r if not self.compute_cumulative else self.trajectory_returns,
                             self.riskfree_rate, self.periods_per_year)


class CalmarEnv(PortfolioOptimizationEnv):
    """
    This class implements a custom environment following the `gym` structure for Portfolio Optimization.
    Using the Calmar ratio as the rewards
    """

    def __init__(self,
                 df_ohlc: pd.DataFrame,
                 df_observations: pd.DataFrame,
                 periods_per_year: int = 252,
                 compute_cumulative: bool = False,
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
        """
        :param df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
        :param df_observations: pd.DataFrame with the environment features.
        :param rebalance_every: timesteps between consecutive rebalancing actions.
        :param continuous_weights: should weights be continuous and consider close prices, or splitted in W_h, W_b, W_s
        :param max_trajectory_len: max length for the trajectories
        :param observation_frame_lookback: return the previous N observations from the environment to take the next action.
        :param slippage: %loss due to gap between reference (Open) price and the execution price.
        :param transaction_costs: %loss due to execution of the trade.
        :param periods_per_year: 252 for daily returns
        :param compute_cumulative: boolean flag to indicate if the reward should consider or not the complete trajectory
        """
        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)

        self.periods_per_year = periods_per_year
        self.compute_cumulative = compute_cumulative

    def compute_reward(self, r):
        """
        :param r: returns series
        :return: calmar ratio of the returns or the full trajectory
        """
        if isinstance(self.trajectory_returns, list):
            trajectory_rets = pd.concat(self.trajectory_returns)
        else:
            trajectory_rets = self.trajectory_returns

        trajectory_rets = torch.Tensor(trajectory_rets).squeeze()

        return calmar_ratio(trajectory_rets, self.periods_per_year)


    def step(self, action):

        _ = super().step(action)

        if len(_) == 4:
            observations, reward, done, info = _
        else:
            observations, reward, truncated, terminated, info = _
            done = truncated or terminated

        print("Original reward:")
        print(reward)
        print(self.trajectory_returns)
        if done:
            reward = 0.0
        else:
            reward = self.compute_reward(self.trajectory_returns)

        if self.convert_to_terminated_truncated:
            return observations, reward, truncated, terminated, info
        else:
            return observations, reward, done, info


class TrackingErrorEnv(PortfolioOptimizationEnv):
    """
     This class implements a custom environment following the `gym` structure for Portfolio Optimization.
     Using the Sharpe ratio as the rewards
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
        """
        :param df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
        :param df_observations: pd.DataFrame with the environment features.
        :param df_reference: pd.DataFrame with the reference index to compute the tracking error.
        :param rebalance_every: timesteps between consecutive rebalancing actions.
        :param continuous_weights: should weights be continuous and consider close prices, or splitted in W_h, W_b, W_s
        :param max_trajectory_len: max length for the trajectories
        :param observation_frame_lookback: return the previous N observations from the environment to take the next action.
        :param slippage: %loss due to gap between reference (Open) price and the execution price.
        :param transaction_costs: %loss due to execution of the trade.
        :param periods_per_year: 252 for daily returns
        :param render_mode: how to output the environment signals
        :param agent_type: has discrete or continuous actions?
        :param convert_to_terminated_truncated: use done (old Gym version) or truncated and terminated (new)
        :param compute_cumulative: boolean flag to indicate if the reward should consider or not the complete trajectory
        """
        self.df_reference = df_reference
        self.reference_returns = None
        self.trajectory_reference_returns = None
        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)

    def reset(self, seed: int = None, options: dict = None):
        """
        Method to reset the environment,
        :param seed: int, random seed.
        :param options: dictionary of options
        :return:
        """
        self.trajectory_reference_returns = []
        obs, info = super().reset()
        return obs, info

    def compute_reward(self, r):
        """
        :param r: returns series
        :return: calmar ratio of the returns or the full trajectory
        """
        if isinstance(self.trajectory_returns, list):
            trajectory_rets = pd.concat(self.trajectory_returns)
        else:
            trajectory_rets = self.trajectory_returns

        trajectory_rets = torch.Tensor(trajectory_rets).squeeze()

        reference_rets =  pd.concat(self.trajectory_reference_returns).drop_duplicates()
        reference_rets = reference_rets.tail(reference_rets.shape[0]-1)
        reference_rets = torch.Tensor(reference_rets).squeeze()

        terror = tracking_error(trajectory_rets, reference_rets)
        creward = torch.clip(torch.pow(terror, -1), 0, 1e5)
        return creward

    def step(self, action):
        self.reference_returns = self.df_reference[self.current_rebalancing_date:self.next_rebalancing_date]
        self.trajectory_reference_returns.append(self.reference_returns)

        _ = super().step(action)

        if len(_) == 4:
            observations, reward, done, info = _
        else:
            observations, reward, truncated, terminated, info = _
            done = truncated or terminated

        reward = self.compute_reward(self.reference_returns)

        if self.convert_to_terminated_truncated:
            return observations, reward, truncated, terminated, info
        else:
            return observations, reward, done, info



class MarketexitEnv(PortfolioOptimizationEnv):
    """
     This class implements a custom environment following the `gym` structure for Portfolio Optimization.
     Using the Sharpe ratio as the rewards
     """

    def __init__(self,
                 df_ohlc: pd.DataFrame,
                 df_observations: pd.DataFrame,
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
                 pos_factor: float = 0.0,
                 neg_factor: float = -1.0
                 ):
        """
        :param df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
        :param df_observations: pd.DataFrame with the environment features.
        :param rebalance_every: timesteps between consecutive rebalancing actions.
        :param continuous_weights: should weights be continuous and consider close prices, or splitted in W_h, W_b, W_s
        :param max_trajectory_len: max length for the trajectories
        :param observation_frame_lookback: return the previous N observations from the environment to take the next action.
        :param slippage: %loss due to gap between reference (Open) price and the execution price.
        :param transaction_costs: %loss due to execution of the trade.
        :param periods_per_year: 252 for daily returns
        :param render_mode: how to output the environment signals
        :param agent_type: has discrete or continuous actions?
        :param convert_to_terminated_truncated: use done (old Gym version) or truncated and terminated (new)
        :param compute_cumulative: boolean flag to indicate if the reward should consider or not the complete trajectory
        :param pos_factor: factor to multiply the positive returns, 0.0 to dampen
        :param neg_factor: factor to multiply the negative returns and make them positive, -1.0 by default.
        """
        self.trajectory_reference_returns = None
        self.pos_factor = pos_factor
        self.neg_factor = neg_factor
        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)

    def reset(self, seed: int = None, options: dict = None):
        """
        Method to reset the environment,
        :param seed: int, random seed.
        :param options: dictionary of options
        :return:
        """
        obs, info = super().reset()
        return obs, info

    def compute_reward(self, r):
        """
        :param r: returns series
        :return: market positioning
        """

        return torch.sum(torch.log(1 + (r.ge(0)*r*self.pos_factor + r.lt(0)*r*self.neg_factor)))


    def step(self, action):
        # We get this to update the weights of the portfolio given the action
        _ = super().step(action)

        if len(_) == 4:
            observations, reward, done, info = _
        else:
            observations, reward, truncated, terminated, info = _
            done = truncated or terminated

        reward = self.compute_reward(self.last_returns)

        if self.convert_to_terminated_truncated:
            return observations, reward, truncated, terminated, info
        else:
            return observations, reward, done, info




class DrawdownEnv(PortfolioOptimizationEnv):
    """
     This class implements a custom environment following the `gym` structure for Portfolio Optimization.
     Using the Sharpe ratio as the rewards
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
                 max_drawdown: float = 0.2,
                 observation_frame_lookback: int = 5,
                 render_mode: str = 'tile',
                 agent_type: str = 'discrete',
                 convert_to_terminated_truncated: bool = False,
                 verbose: int = 0
                 ):
        """
        :param df_ohlc: pd.DataFrame with the OHLC+ of the portfolio instruments. Columns are multiindex (Symbol, Price) e.g. ('AAPL', 'Close')
        :param df_observations: pd.DataFrame with the environment features.
        :param df_reference: pd.DataFrame with the reference index to compute the tracking error.
        :param rebalance_every: timesteps between consecutive rebalancing actions.
        :param continuous_weights: should weights be continuous and consider close prices, or splitted in W_h, W_b, W_s
        :param max_trajectory_len: max length for the trajectories
        :param observation_frame_lookback: return the previous N observations from the environment to take the next action.
        :param slippage: %loss due to gap between reference (Open) price and the execution price.
        :param transaction_costs: %loss due to execution of the trade.
        :param periods_per_year: 252 for daily returns
        :param render_mode: how to output the environment signals
        :param agent_type: has discrete or continuous actions?
        :param convert_to_terminated_truncated: use done (old Gym version) or truncated and terminated (new)
        :param compute_cumulative: boolean flag to indicate if the reward should consider or not the complete trajectory
        """
        self.df_reference = df_reference
        self.reference_returns = None
        self.trajectory_reference_returns = None
        self.max_drawdown = max_drawdown

        super().__init__(df_ohlc, df_observations, rebalance_every, slippage, transaction_costs,
                         continuous_weights, allow_short_positions, max_trajectory_len, observation_frame_lookback,
                         render_mode, agent_type, convert_to_terminated_truncated, verbose)

    def reset(self, seed: int = None, options: dict = None):
        """
        Method to reset the environment,
        :param seed: int, random seed.
        :param options: dictionary of options
        :return:
        """
        self.trajectory_reference_returns = []
        obs, info = super().reset()
        return obs, info

    def compute_reward(self, r):
        """
        :param r: returns series
        :return: calmar ratio of the returns or the full trajectory
        """
        if isinstance(self.trajectory_returns, list):
            trajectory_rets = pd.concat(self.trajectory_returns)
        else:
            trajectory_rets = self.trajectory_returns

        trajectory_rets = torch.Tensor(trajectory_rets).squeeze()


        terror = tracking_error(trajectory_rets, reference_rets)
        creward = torch.clip(torch.pow(terror, -1), 0, 1e5)

        return creward

    def step(self, action):
        self.reference_returns = self.df_reference[self.current_rebalancing_date:self.next_rebalancing_date]
        self.trajectory_reference_returns.append(self.reference_returns)


        _ = super().step(action)

        if len(_) == 4:
            observations, reward, done, info = _
        else:
            observations, reward, truncated, terminated, info = _
            done = truncated or terminated

        #ToDo check if max admissible drawdown has been achieved

        reward = self.compute_reward(self.reference_returns)

        if self.convert_to_terminated_truncated:
            return observations, reward, truncated, terminated, info
        else:
            return observations, reward, done, info