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
import torch


def get_price_window(df: pd.DataFrame, ref_date: str, lookback_window: int) -> pd.DataFrame:
    """
    Creates a price window given a history dataframe, using the last reference date and a lookback window.
    :param df: dataframe of prices
    :param ref_date: date of the last available prices
    :param int lookback_window: periods to look back in history to get the window. Negative values to take all available history.
    :return: Last lookback_window observations of a table.
    :rtype: pd.DataFrame
    """
    _ = df[df.index <= ref_date].sort_index()
    return _ if lookback_window < 0 else _.tail(lookback_window)


def decompose_weights_df(df_weights: pd.DataFrame) -> tuple:
    """
    This function takes a dataframe of portfolio weights and decomposes in 3 matrices:
    W_h, W_s, W_b. Satisfying this equation:
    W_h(t) = W_n where W_n==W_o
    W_s = W_h(t)-W_h(t-1) <0 : negative weight delta, necessary sales
    W_b = W_h(t)-W_h(t-1) >0 : positive weight delta, necessary buys

    :param pd.DataFrame df_weights: weights dataframe
    :return: The weights of the instruments to hold, buy or sale in that order
    :rtype: tuple
    """
    deltas = df_weights.diff()
    w_b = np.abs(deltas * (deltas >= 0))
    w_s = deltas * (deltas <= 0)
    w_h = df_weights * (df_weights <= df_weights.shift(1)) + df_weights.shift(1) * (df_weights > df_weights.shift(1))
    return w_h, w_b, w_s


def decompose_weights_tensor(w_n: torch.Tensor, w_o: torch.Tensor) -> tuple:
    """
    This function takes the old and new tensor of weights in the portfolio and decomposes
    it in the (h)eld, (b)ought and (sold) assets W_h, W_s, W_b. Satisfying this equation:
    W_h(t) = W_n where W_n==W_o
    W_s = W_h(t)-W_h(t-1) <0 : negative weight delta, necessary sales
    W_b = W_h(t)-W_h(t-1) >0 : positive weight delta, necessary buys

    :param w_n: new weight positions
    :param w_o: previous weight positions
    :return: w_h (Weights of instruments to hold), w_b (Necessary buys), w_s (Necessary sales)
    :rtype: tuple
    """
    w_delta = w_n - w_o
    w_b = torch.abs(w_delta * (w_delta > 0)).float()
    w_s = torch.abs(w_delta * (w_delta < 0)).float()
    w_h = 0.5*(w_o+w_n-w_s-w_b).float()
    assert torch.max(torch.abs(w_h+w_b-w_n))<1e-6
    return w_h, w_b, w_s


def compute_OHLC_returns(df_ohlc: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes an OHLC matrix of prices and computes the daily returns for each of the available actions.
    For daily returns:
    (Buy) action has a daily return of Close_t/Open_t
    (Sell) action has a daily return of the overnight change Open_t/Close_t-1
    (Hold) action has a daily return of Close_t/Close_t-1

    For other periodicitie:
    (Buy) action has a period-over-period return of Close_t/Open_t, assuming that it is immediately bought at the beginning of period t.
    (Sell) action has a period-over-period return Open_t/Close_t-1, assuming that it is immediately sold at the beginning of period t.
    (Hold) action has a period-over-period return of Close_t/Close_t-1.

    :param pd.DataFrame df_ohlc: OHLC returns of the assets
    :return: Dataframe with the returns of the held/bought/sold assets.
    :rtype: pd.DataFrame
    """
    df_rets = df_ohlc.copy(deep = True)
    df_rets['r_b'] = (df_rets['Close'] / df_rets['Open']) - 1
    df_rets['r_s'] = (df_rets['Open'] / df_rets['Close'].shift(1)) - 1
    df_rets['r_h'] = (df_rets['Close'] / df_rets['Close'].shift(1)) - 1

    return df_rets.loc[:, ['r_h', 'r_b', 'r_s']]


def create_return_matrices(df_instruments: pd.DataFrame) -> tuple:
    """
    This function takes a dataframe of OHLC instruments and computes the return matrices Rh, Rb and Rs for each action
    (Buy) action has a daily return of Close_t/Open_t
    (Sell) action has a daily return of the overnight change Open_t/Close_t-1
    (Hold) action has a daily return of Close_t/Close_t-1
    :param df_instruments: dataframe with OHLC of the data and column index [i,j] where i is the instrument and j the OHLC prices
    :return: typle with the return dataframes of held, bought and sold assets
    :rtype: tuple
    """
    r_h = dict()
    r_b = dict()
    r_s = dict()
    for i in set([i[0] for i in df_instruments.columns]):
        _ = compute_OHLC_returns(df_instruments[i])
        r_h[i] = _['r_h'].values
        r_b[i] = _['r_b'].values
        r_s[i] = _['r_s'].values

    r_h = pd.DataFrame(r_h, index=df_instruments.index).dropna()
    r_b = pd.DataFrame(r_b, index=df_instruments.index).dropna()
    r_s = pd.DataFrame(r_s, index=df_instruments.index).dropna()

    return r_h, r_b, r_s


def normalize_price_window(df_price_window: pd.DataFrame, norm_pivot: str = 'window_close') -> pd.DataFrame:
    """
    This function normalizes an OHLC dataframe of N days based on a criterion. window_close uses the last closing price,
    window_open the first open price. prev_day_close / day_open the first or last price available for the day.
    Normalization is done as P/normalization reference.
    :param df_price_window: OHLC prices
    :param norm_pivot: name of the normalization strategy. One of window_close(open), day_close(open).
    :return: normalized df_price_window.
    :rtype: pd.DataFrame
    """
    if norm_pivot == 'prev_day_close':
        df_norm_price = df_price_window.div(df_price_window['Close'].shift(1), axis=0)
    elif norm_pivot == 'day_close':
        df_norm_price = df_price_window.div(df_price_window['Close'], axis=0)
    elif norm_pivot == 'day_open':
        df_norm_price = df_price_window.div(df_price_window['Open'], axis=0)
    elif norm_pivot == 'window_close':
        df_norm_price = df_price_window / df_price_window['Close'][-1]
    elif norm_pivot == 'window_open':
        df_norm_price = df_price_window / df_price_window['Open'][0]
    else:
        raise ValueError(f"{norm_pivot} is not a valid normalization strategy")

    return df_norm_price



