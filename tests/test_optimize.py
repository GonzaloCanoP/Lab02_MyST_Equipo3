import pandas as pd
import numpy as np
import optuna
import pytest

from src.optimize import (
    cost_sweep,
    diagnostic_study,
    search_space,
    select_plateau,
    sensitivity,
    single_indicator_comparison,
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

def test_single_indicator_comparison(
    monkeypatch,
    config_test,
):
    class FakeResult:
        def __init__(
            self,
            n_trades,
        ):
            self.trades = (
                pd.DataFrame(
                    {
                        "pnl_net":
                            [1.0]
                            * n_trades
                    }
                )
            )

    def fake_evaluate_params(
        params_by_regime,
        prices,
        regimes,
        config,
        entry_mask=None,
    ):
        return (
            FakeResult(40),
            {
                "calmar": 1.5
            },
        )

    scores = {
        "sma": (30, 0.8),
        "macd": (25, 0.6),
        "rsi": (20, 0.4),
    }

    def fake_single_indicator(
        indicator,
        params_by_regime,
        prices,
        regimes,
        config,
    ):
        n_trades, calmar = (
            scores[indicator]
        )

        return (
            FakeResult(n_trades),
            {
                "calmar": calmar
            },
        )

    monkeypatch.setattr(
        "src.optimize._evaluate_params",
        fake_evaluate_params,
    )

    monkeypatch.setattr(
        "src.optimize._evaluate_single_indicator",
        fake_single_indicator,
    )

    result = (
        single_indicator_comparison(
            {
                "tendencia":
                    config_test[
                        "base_params"
                    ]
            },
            None,
            None,
            config_test,
        )
    )

    assert set(
        result.index
    ) == {
        "2_de_3",
        "sma",
        "macd",
        "rsi",
    }

    assert (
        result.loc[
            "2_de_3",
            "n_trades",
        ]
        == 40
    )

    assert result.loc[
        "2_de_3",
        "calmar",
    ] == pytest.approx(
        1.5
    )

    assert (
        result.loc[
            "sma",
            "n_trades",
        ]
        == 30
    )

    assert result.loc[
        "macd",
        "calmar",
    ] == pytest.approx(
        0.6
    )

    assert result.loc[
        "rsi",
        "calmar",
    ] == pytest.approx(
        0.4
    )

def test_cost_sweep_finds_breakeven(
    monkeypatch,
    config_test,
):
    class FakeResult:
        def __init__(self):
            self.trades = pd.DataFrame(
                {
                    "pnl_net": [
                        1.0,
                        -1.0,
                    ]
                }
            )

    original_commission = (
        config_test[
            "commission"
        ]
    )

    original_slippage = (
        config_test[
            "slippage"
        ]
    )

    def fake_evaluate(
        params_by_regime,
        prices,
        regimes,
        config,
        entry_mask=None,
    ):
        round_trip = (
            2.0
            * (
                config[
                    "commission"
                ]
                + config[
                    "slippage"
                ]
            )
            * 10_000.0
        )

        net_return = (
            0.05
            - round_trip
            / 1000.0
        )

        return (
            FakeResult(),
            {
                "ann_return":
                    net_return,
                "calmar":
                    1.0,
            },
        )

    monkeypatch.setattr(
        "src.optimize._evaluate_params",
        fake_evaluate,
    )

    result = cost_sweep(
        {
            "tendencia":
                config_test[
                    "base_params"
                ]
        },
        None,
        None,
        config_test,
        [
            100,
            0,
            50,
            25,
            75,
        ],
    )

    assert list(
        result.index
    ) == [
        0.0,
        25.0,
        50.0,
        75.0,
        100.0,
    ]

    assert result.loc[
        0.0,
        "net_return",
    ] == pytest.approx(
        0.05
    )

    assert result.loc[
        50.0,
        "net_return",
    ] == pytest.approx(
        0.0
    )

    assert result[
        "break_even_bps"
    ].iloc[0] == pytest.approx(
        50.0
    )

    assert result[
        "base_cost_bps"
    ].iloc[0] == pytest.approx(
        29.0
    )

    assert result[
        "margin_bps"
    ].iloc[0] == pytest.approx(
        21.0
    )

    assert (
        config_test[
            "commission"
        ]
        == original_commission
    )

    assert (
        config_test[
            "slippage"
        ]
        == original_slippage
    )

def test_cost_sweep_rejects_negative_cost(
    config_test,
):
    with pytest.raises(
        ValueError
    ):
        cost_sweep(
            {
                "tendencia":
                    config_test[
                        "base_params"
                    ]
            },
            None,
            None,
            config_test,
            [
                0,
                -5,
                10,
            ],
        )

def _diagnostic_prices():
    """Precios mínimos para probar la lógica del estudio sin correr backtests reales."""
    index = pd.bdate_range(
        "2019-12-02",
        "2020-07-31",
    )

    return {
        "AAPL": pd.DataFrame(
            index=index
        )
    }


@pytest.mark.parametrize(
    "sampler_name,sampler_type",
    [
        (
            "random",
            optuna.samplers.RandomSampler,
        ),
        (
            "tpe",
            optuna.samplers.TPESampler,
        ),
    ],
)
def test_diagnostic_study_runs_requested_trials(
    monkeypatch,
    config_test,
    sampler_name,
    sampler_type,
):
    prices = _diagnostic_prices()

    config = dict(
        config_test
    )

    config["blocks"] = {
        "train": (
            "2020-01-01",
            "2020-06-30",
        )
    }

    config[
        "embargo_days"
    ] = 5

    config[
        "min_trades_per_window"
    ] = 24

    seen_masks = []

    class FakeResult:
        def __init__(self):
            self.trades = pd.DataFrame(
                index=range(30)
            )

    def fake_evaluate(
        params_by_regime,
        prices,
        regimes,
        config,
        entry_mask=None,
    ):
        seen_masks.append(
            entry_mask.copy()
        )

        return (
            FakeResult(),
            {
                "calmar": 1.25
            },
        )

    monkeypatch.setattr(
        "src.optimize._evaluate_params",
        fake_evaluate,
    )

    study = diagnostic_study(
        prices,
        config,
        sampler=sampler_name,
        n_trials=3,
        seed=42,
    )

    assert isinstance(
        study.sampler,
        sampler_type,
    )

    assert len(
        study.trials
    ) == 3

    assert study.user_attrs[
        "n_trials_requested"
    ] == 3

    assert study.user_attrs[
        "n_trials_evaluated"
    ] == 3

    assert study.user_attrs[
        "n_trials_feasible"
    ] == 3

    assert study.user_attrs[
        "elapsed_seconds"
    ] >= 0.0

    mask = seen_masks[0]

    train_mask = mask.loc[
        "2020-01-01":
        "2020-06-30"
    ]

    assert train_mask.iloc[
        :-5
    ].all()

    assert not train_mask.iloc[
        -5:
    ].any()

    assert not mask.loc[
        :"2019-12-31"
    ].any()

def test_diagnostic_study_rejects_low_activity(
    monkeypatch,
    config_test,
):
    prices = _diagnostic_prices()

    config = dict(
        config_test
    )

    config["blocks"] = {
        "train": (
            "2020-01-01",
            "2020-06-30",
        )
    }

    config[
        "embargo_days"
    ] = 5

    config[
        "min_trades_per_window"
    ] = 24

    class FakeResult:
        def __init__(self):
            self.trades = pd.DataFrame(
                index=range(23)
            )

    def fake_evaluate(
        params_by_regime,
        prices,
        regimes,
        config,
        entry_mask=None,
    ):
        return (
            FakeResult(),
            {
                "calmar": 10.0
            },
        )

    monkeypatch.setattr(
        "src.optimize._evaluate_params",
        fake_evaluate,
    )

    study = diagnostic_study(
        prices,
        config,
        sampler="random",
        n_trials=1,
        seed=42,
    )

    trial = study.trials[0]

    assert trial.value == -np.inf

    assert (
        trial.user_attrs[
            "feasible"
        ]
        is False
    )

    assert trial.user_attrs[
        "n_trades"
    ] == 23