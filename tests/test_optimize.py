import numpy as np
import optuna
import pytest

from src.optimize import (
    search_space,
    select_plateau,
)


def test_search_space_matches_config(
    config_test,
):
    space = search_space()

    assert set(space) == set(
        config_test[
            "search_ranges"
        ]
    )

    integer_parameters = {
        "sma_fast",
        "sma_slow",
        "rsi_window",
        "rsi_lo",
        "rsi_hi",
        "max_holding",
    }

    for name, kind in space.items():
        low, high = config_test[
            "search_ranges"
        ][name]

        assert low < high

        if name in integer_parameters:
            assert kind == "int"
        else:
            assert kind == "float"


def test_select_plateau_returns_medoid():
    study = optuna.create_study(
        direction="maximize"
    )

    distributions = {
        "sma_fast":
            optuna.distributions.IntDistribution(
                5,
                30,
            ),
        "k_stop":
            optuna.distributions.FloatDistribution(
                1.0,
                4.0,
            ),
    }

    trials = [
        (
            {
                "sma_fast": 5,
                "k_stop": 1.0,
            },
            10.0,
        ),
        (
            {
                "sma_fast": 15,
                "k_stop": 2.0,
            },
            9.5,
        ),
        (
            {
                "sma_fast": 30,
                "k_stop": 4.0,
            },
            9.4,
        ),
        (
            {
                "sma_fast": 30,
                "k_stop": 1.0,
            },
            1.0,
        ),
    ]

    for params, value in trials:
        study.add_trial(
            optuna.trial.create_trial(
                params=params,
                distributions=distributions,
                value=value,
            )
        )

    study.add_trial(
        optuna.trial.create_trial(
            params={
                "sma_fast": 10,
                "k_stop": 3.0,
            },
            distributions=distributions,
            value=-np.inf,
        )
    )

    result = select_plateau(
        study,
        top_frac=0.75,
    )

    assert result == {
        "sma_fast": 15,
        "k_stop": 2.0,
    }


def test_select_plateau_rejects_invalid_fraction():
    study = optuna.create_study(
        direction="maximize"
    )

    with pytest.raises(
        ValueError
    ):
        select_plateau(
            study,
            top_frac=0.0,
        )