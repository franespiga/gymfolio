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

import torch

def annualize_rets(r: torch.Tensor, periods_per_year: int = 252) -> torch.Tensor:
    """
    Calculates the annualized return of a return series by assuming the returns compound.
    :param r: Tensor of returns, typically daily or monthly returns.
    :param int periods_per_year: Number of periods in a year (default is 252 for daily returns).
    :return: Annualized return.
    :rtype: torch.Tensor

    Explanation: This function takes a series of periodic returns and annualizes them by
    compounding over the specified periods per year. It uses the formula:
    Annualized Return = (1 + Total Return)^(Periods per Year / Number of Periods) - 1
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r: torch.Tensor, periods_per_year: int = 252) -> torch.Tensor:
    """
    Calculates the annualized volatility (standard deviation) of returns.
    :param r: Tensor of returns.
    :param int periods_per_year: Number of periods per year (252 by default for daily returns).
    :return: Annualized volatility.
    :rtype: torch.Tensor

    Explanation: This function computes the annualized standard deviation by multiplying the
    periodic standard deviation by the square root of the periods per year:
    Annualized Volatility = Periodic Std Dev * sqrt(Periods per Year)
    """
    return torch.std(r, unbiased=True) * (periods_per_year ** 0.5)


def sharpe_ratio(r: torch.Tensor, riskfree_rate: float, periods_per_year: int = 252) -> torch.Tensor:
    """
    Computes the annualized Sharpe ratio of a return series.
    :param r: Tensor of returns.
    :param riskfree_rate: Annual risk-free rate as a decimal.
    :param int periods_per_year: Number of periods per year (default 252).
    :return: Sharpe ratio.
    :rtype: torch.Tensor

    Explanation: The Sharpe ratio measures the excess return (above risk-free rate) per unit
    of volatility. It uses annualized excess return divided by annualized volatility:
    Sharpe Ratio = Annualized Excess Return / Annualized Volatility
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def sortino_ratio(r: torch.Tensor, riskfree_rate: float, periods_per_year: int = 252) -> torch.Tensor:
    """
    Computes the annualized Sortino ratio, which uses downside volatility only.
    :param r: Tensor of returns.
    :param float riskfree_rate: Annual risk-free rate.
    :param int periods_per_year: 252 by default for daily returns.
    :return: Sortino ratio.
    :rtype: torch.Tensor

    Explanation: The Sortino ratio is a variation of the Sharpe ratio focusing only on downside risk
    (negative returns). It uses semi-deviation for risk calculation instead of full volatility.
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(torch.masked_select(r, r < 0), periods_per_year)
    return ann_ex_ret / ann_vol


def tracking_error(r: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Calculates the Tracking Error between the portfolio return and a reference benchmark.
    :param r: Portfolio return series.
    :param ref: Benchmark return series.
    :return: Tracking error, measuring divergence from benchmark.
    :rtype: torch.Tensor

    Explanation: Tracking error quantifies the consistency of returns relative to a benchmark
    by calculating the standard deviation of the difference (r - ref).
    """
    t_e = torch.std(torch.sub(r, ref, alpha=1), unbiased=True)
    if t_e.isnan():
        return -torch.std(torch.sub(r, ref, alpha=1), unbiased=False)
    else:
        return t_e


def calmar_ratio(r: torch.Tensor, periods_per_year: int = 252) -> torch.Tensor:
    """
    Calculates the Calmar ratio, which is the annualized return divided by the maximum drawdown.
    :param r: Tensor of returns.
    :param int periods_per_year: Number of periods per year, 252 by default.
    :return: Calmar ratio.
    :rtype: torch.Tensor

    Explanation: The Calmar ratio assesses return relative to risk by comparing the
    annualized return to the maximum observed drawdown, highlighting performance during downturns.
    """
    cumulative_returns = torch.cumprod(1 + r, 0)
    cumulative_max = torch.cummax(cumulative_returns, 0).values
    drawdowns = (cumulative_max - cumulative_returns) / cumulative_max
    max_dd = torch.max(drawdowns)

    return annualize_rets(r, periods_per_year) / max_dd
