import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    compute_metrics,
    drawdown_series,
)


def test_drawdown_series_matches_manual_calculation():
    equity = pd.Series(
        [
            100.0,
            105.0,
            102.0,
            110.0,
            108.0,
            115.0,
            111.0,
            118.0,
            116.0,
            120.0,
        ],
        index=pd.bdate_range(
            "2020-01-01",
            periods=10,
        ),
    )

    expected = (
        equity
        / equity.cummax()
        - 1
    ).rename("drawdown")

    result = drawdown_series(
        equity
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_compute_metrics_matches_manual_calculation():
    equity = pd.Series(
        [
            100.0,
            105.0,
            102.0,
            110.0,
            108.0,
            115.0,
            111.0,
            118.0,
            116.0,
            120.0,
        ],
        index=pd.bdate_range(
            "2020-01-01",
            periods=10,
        ),
    )

    trades = pd.DataFrame(
        {
            "pnl_net": [
                100.0,
                -50.0,
                200.0,
                -100.0,
            ]
        }
    )

    periods = 252

    returns = equity.pct_change(
        fill_method=None
    ).dropna()

    expected_ann_return = (
        equity.iloc[-1]
        / equity.iloc[0]
    ) ** (
        periods / len(returns)
    ) - 1

    expected_ann_vol = (
        returns.std(ddof=1)
        * np.sqrt(periods)
    )

    expected_sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(periods)
    )

    downside = np.minimum(
        returns.to_numpy(
            dtype=float
        ),
        0.0,
    )

    downside_deviation = np.sqrt(
        np.mean(downside**2)
    )

    expected_sortino = (
        returns.mean()
        / downside_deviation
        * np.sqrt(periods)
    )

    expected_mdd = (
        equity
        / equity.cummax()
        - 1
    ).min()

    expected_calmar = (
        expected_ann_return
        / abs(expected_mdd)
    )

    result = compute_metrics(
        equity,
        trades,
        rf=0.0,
        periods_per_year=periods,
    )

    assert result[
        "ann_return"
    ] == pytest.approx(
        expected_ann_return,
        abs=1e-9,
    )

    assert result[
        "ann_vol"
    ] == pytest.approx(
        expected_ann_vol,
        abs=1e-9,
    )

    assert result[
        "sharpe"
    ] == pytest.approx(
        expected_sharpe,
        abs=1e-9,
    )

    assert result[
        "sortino"
    ] == pytest.approx(
        expected_sortino,
        abs=1e-9,
    )

    assert result[
        "max_drawdown"
    ] == pytest.approx(
        expected_mdd,
        abs=1e-9,
    )

    assert result[
        "calmar"
    ] == pytest.approx(
        expected_calmar,
        abs=1e-9,
    )

    assert result[
        "n_trades"
    ] == 4

    assert result[
        "win_rate"
    ] == pytest.approx(
        0.5
    )

    assert result[
        "payoff_ratio"
    ] == pytest.approx(
        2.0
    )