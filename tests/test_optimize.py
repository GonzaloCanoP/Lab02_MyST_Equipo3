import numpy as np
import optuna
import pytest

from src.optimize import (
    search_space,
    select_plateau,
    sensitivity,
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

def test_sensitivity_varies_each_parameter(
    monkeypatch,
    config_test,
):
    params = {
        "tendencia": dict(
            config_test[
                "base_params"
            ]
        )
    }

    regimes = None
    prices = None

    def fake_evaluate(
        params_by_regime,
        prices,
        regimes,
        config,
        entry_mask=None,
    ):
        score = sum(
            float(value)
            for regime_params
            in params_by_regime.values()
            for value
            in regime_params.values()
        )

        return None, {
            "calmar": score
        }

    monkeypatch.setattr(
        "src.optimize._evaluate_params",
        fake_evaluate,
    )

    result = sensitivity(
        params,
        prices,
        regimes,
        config_test,
        pct=0.20,
    )

    assert len(result) == 18

    assert set(
        result["factor"]
    ) == {
        0.8,
        1.2,
    }

    sma_down = result[
        (
            result["parameter"]
            == "sma_fast"
        )
        & (
            result["factor"]
            == 0.8
        )
    ].iloc[0]

    sma_up = result[
        (
            result["parameter"]
            == "sma_fast"
        )
        & (
            result["factor"]
            == 1.2
        )
    ].iloc[0]

    assert (
        sma_down[
            "tested_value"
        ]
        == 16
    )

    assert (
        sma_up[
            "tested_value"
        ]
        == 24
    )

    risk_down = result[
        (
            result["parameter"]
            == "risk_per_trade"
        )
        & (
            result["factor"]
            == 0.8
        )
    ].iloc[0]

    assert risk_down[
        "tested_value"
    ] == pytest.approx(
        0.008
    )

def test_sensitivity_rejects_invalid_pct(
    config_test,
):
    with pytest.raises(
        ValueError
    ):
        sensitivity(
            {
                "tendencia":
                    config_test[
                        "base_params"
                    ]
            },
            None,
            None,
            config_test,
            pct=0.0,
        )