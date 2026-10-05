import numpy as np
import pandas as pd

from src.signals import (
    compute_indicators,
    indicator_votes,
    confirm_signal,
    signal_strength,
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