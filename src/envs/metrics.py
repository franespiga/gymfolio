# METRICS
import torch

def annualize_rets(r: torch.Tensor, periods_per_year: int = 252) -> torch.Tensor:
    """
    Returns the annualized returns from a series of returns
    :param r: return series
    :param int periods_per_year: number of periods per year, 252 by default for daily returns
    :return: annualized returns
    :rtype: torch.Tensor
    """
    compounded_growth = (1 + r).prod()
    n_periods = r.shape[0]
    return compounded_growth ** (periods_per_year / n_periods) - 1


def annualize_vol(r: torch.Tensor, periods_per_year: int = 252) -> torch.Tensor:
    """
    Annualizes the volatility of a set of returns
    :param r: return series
    :param int periods_per_year: 252 by default for daily returns
    :return: annualized volatility of the returns
    :rtype: torch.Tensor
    """
    return torch.std(r, unbiased=True) * (periods_per_year ** 0.5)


def sharpe_ratio(r: torch.Tensor, riskfree_rate:float, periods_per_year:int = 252) -> torch.Tensor:
    """
    Computes the annualized Sharpe ratio of a set of returns
    :param r: return series
    :param riskfree_rate: risk free rate
    :param int periods_per_year: number of periods per year, 252 by default for daily returns
    :return: Sharpe ratio of the returns series
    :rtype: torch.Tensor
    """
    # convert the annual riskfree rate to per period
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol


def sortino_ratio(r: torch.Tensor, riskfree_rate: float, periods_per_year: int = 252) -> torch.Tensor:
    """
    Computes the annualized sortino ratio of a set of returns using the semideviation
    :param r: return series
    :param float riskfree_rate: risk free rate
    :param int periods_per_year: number of periods per year, 252 by default for daily returns
    :return: the Sortino ratio of the returns series
    :rtype: torch.Tensor
    """
    rf_per_period = (1 + riskfree_rate) ** (1 / periods_per_year) - 1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(torch.masked_select(r, r < 0), periods_per_year)
    return ann_ex_ret / ann_vol


def tracking_error(r, ref) -> torch.Tensor:
    """
    Returns the Tracking Error of a series of returns r and a reference series of returns ref
    :param r: return series
    :param ref: reference (tracked index) returns
    :return: Tracking error between the returns series and reference returns
    :rtype: torch.Tensor
    """
    t_e = torch.std(torch.sub(r, ref, alpha=1), unbiased = True)
    if t_e.isnan():
        return torch.std(torch.sub(r, ref, alpha=1), unbiased = False)
    else:
        return t_e

def calmar_ratio(r, periods_per_year: int = 252) -> torch.Tensor:
    """
    Computes the Calmar ratio (annualized returns / maximum drawdown)
    :param r: return series
    :param int periods_per_year: number of periods per year, 252 by default for daily returns
    :return: Calmar ratio of the returns series.
    :rtype: torch.Tensor
    """
    cumulative_returns = torch.cumprod(1+r, 0)
    cumulative_max = torch.cummax(cumulative_returns, 0).values
    drawdowns = (cumulative_max - cumulative_returns) / cumulative_max
    max_dd = torch.max(drawdowns)

    return annualize_rets(r, periods_per_year) / max_dd

