import numpy as np
import pandas as pd
import pytest

from src.signals import (
    compute_indicators,
    indicator_votes,
    confirm_signal,
    signal_strength,
    generate_signals,
    sma_macd_vote_correlation,
)

def test_confirmation_rule():

    votes = pd.DataFrame(
        {
            "v_sma": [1, 1, 1, -1],
            "v_macd": [0, 1, 1, -1],
            "v_rsi": [0, 0, -1, -1],
        }
    )

    state = confirm_signal(
        votes,
        min_agree=2,
    )

    strength = signal_strength(
        votes,
        min_agree=2,
    )

    assert state.tolist() == [
        0,
        1,
        0,
        -1,
    ]

    np.testing.assert_allclose(
        strength.to_numpy(),
        [
            0.0,
            2 / 3,
            0.0,
            -1.0,
        ],
    )


def test_signal_calculation_is_causal(
    synthetic_prices,
    config_test,
):

    df = synthetic_prices["A0"]

    params = config_test[
        "base_params"
    ]

    full_indicators = (
        compute_indicators(
            df,
            params,
            config_test,
        )
    )

    full_votes = indicator_votes(
        full_indicators,
        params,
    )

    full_state = confirm_signal(
        full_votes,
        config_test["min_agree"],
    )

    full_strength = signal_strength(
        full_votes,
        config_test["min_agree"],
    )

    test_points = [
        150,
        400,
        len(df) - 1,
    ]

    for t in test_points:

        truncated = df.iloc[
            : t + 1
        ]

        indicators = (
            compute_indicators(
                truncated,
                params,
                config_test,
            )
        )

        votes = indicator_votes(
            indicators,
            params,
        )

        state = confirm_signal(
            votes,
            config_test["min_agree"],
        )

        strength = signal_strength(
            votes,
            config_test["min_agree"],
        )

        date = df.index[t]

        pd.testing.assert_series_equal(
            indicators.loc[date],
            full_indicators.loc[date],
            check_names=False,
        )

        pd.testing.assert_series_equal(
            votes.loc[date],
            full_votes.loc[date],
            check_names=False,
        )

        assert (
            state.loc[date]
            == full_state.loc[date]
        )

        assert (
            strength.loc[date]
            == full_strength.loc[date]
        )


def test_no_signal_without_regime(
    synthetic_prices,
    config_test,
):
    """Sin etiqueta de régimen (NaN, antes del primer ajuste de P3) no se abre posición."""

    dates = synthetic_prices["A0"].index

    n_missing = 200

    regimes = pd.Series(
        "tendencia",
        index=dates,
        name="regime",
        dtype=object,
    )

    regimes.iloc[:n_missing] = np.nan

    params = config_test["base_params"]

    params_by_regime = {
        "tendencia": params,
        "reversion": params,
        "crisis": params,
    }

    signals = generate_signals(
        synthetic_prices,
        params_by_regime,
        regimes,
        config_test,
    )

    missing = dates[:n_missing]

    assert (
        signals["state"].loc[missing] == 0
    ).all().all()

    assert (
        signals["strength"].loc[missing] == 0
    ).all().all()

    # Con base_params, a partir del día 200 ya hay historia suficiente: debe haber señales.
    assert (
        signals["state"].iloc[n_missing:] != 0
    ).any().any()

def test_sma_macd_vote_correlation(
    monkeypatch,
    config_test,
):
    dates = pd.bdate_range(
        "2020-01-01",
        periods=4,
    )

    prices = {
        "A": pd.DataFrame(
            {
                "marker": [
                    1,
                    1,
                    1,
                    1,
                ]
            },
            index=dates,
        ),
        "B": pd.DataFrame(
            {
                "marker": [
                    2,
                    2,
                    2,
                    2,
                ]
            },
            index=dates,
        ),
    }

    def fake_compute_indicators(
        ohlcv,
        params,
        config,
    ):
        return pd.DataFrame(
            {
                "sma_slow":
                    1.0,
                "macd_hist":
                    1.0,
                "marker":
                    ohlcv[
                        "marker"
                    ],
            },
            index=ohlcv.index,
        )

    def fake_indicator_votes(
        indicators,
        params,
    ):
        marker = int(
            indicators[
                "marker"
            ].iloc[0]
        )

        if marker == 1:
            sma = [
                1,
                1,
                -1,
                -1,
            ]

            macd = [
                1,
                1,
                -1,
                -1,
            ]

        else:
            sma = [
                1,
                -1,
                1,
                -1,
            ]

            macd = [
                -1,
                1,
                -1,
                1,
            ]

        return pd.DataFrame(
            {
                "v_sma":
                    sma,
                "v_macd":
                    macd,
                "v_rsi":
                    0,
            },
            index=indicators.index,
        )

    monkeypatch.setattr(
        "src.signals.compute_indicators",
        fake_compute_indicators,
    )

    monkeypatch.setattr(
        "src.signals.indicator_votes",
        fake_indicator_votes,
    )

    result = (
        sma_macd_vote_correlation(
            prices,
            config_test[
                "base_params"
            ],
            config_test,
            (
                "2020-01-01",
                "2020-01-31",
            ),
        )
    )

    assert result.loc[
        "A",
        "correlation",
    ] == pytest.approx(
        1.0
    )

    assert result.loc[
        "B",
        "correlation",
    ] == pytest.approx(
        -1.0
    )

    assert result.loc[
        "pooled",
        "correlation",
    ] == pytest.approx(
        0.0
    )

    assert result.loc[
        "A",
        "n_obs",
    ] == 4

    assert result.loc[
        "pooled",
        "n_obs",
    ] == 8