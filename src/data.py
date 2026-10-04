"""Carga, descarga y auditoría de datos del portafolio (P1).

`download_prices` es la única función del proyecto que usa red (CLAUDE.md, sección 3).
"""

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _ticker_filename(ticker: str) -> str:
    """Nombre de archivo de un ticker: sin `^` (`^IRX` → `IRX.csv`). CLAUDE.md, sección 3."""
    return f"{ticker.lstrip('^')}.csv"


def download_prices(tickers: list[str], start: str, end: str, out_dir: str = "data") -> None:
    """Descarga OHLCV diario de Yahoo Finance y guarda un CSV por ticker.

    SPEC punto 1: barras diarias ajustadas por splits y dividendos (`auto_adjust=True`).
    No rellena huecos: lo que Yahoo no entrega queda ausente y se reporta en `audit_prices`.

    Parameters
    ----------
    tickers : list[str]
        Símbolos de Yahoo Finance, por ejemplo `["AAPL", "^IRX"]`.
    start, end : str
        Fechas `YYYY-MM-DD`, ambas INCLUSIVAS (Yahoo trata `end` como exclusiva; aquí se compensa).
    out_dir : str
        Carpeta destino. Cada ticker se guarda como `<out_dir>/<TICKER>.csv`.

    Raises
    ------
    ValueError
        Si Yahoo no devuelve datos para algún ticker.
    """
    import yfinance as yf  # solo esta función depende de la red

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for ticker in tickers:
        raw = yf.download(
            ticker,
            start=start,
            end=end_exclusive,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
        if raw.empty:
            raise ValueError(f"Yahoo Finance no devolvió datos para {ticker}")
        df = raw.rename(columns=str.lower)[OHLCV_COLUMNS]
        df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        df.to_csv(out_path / _ticker_filename(ticker))


def load_prices(tickers: list[str], data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Lee los CSV de `data/` y devuelve un DataFrame OHLCV por ticker con índice común.

    CLAUDE.md, sección 3: intersección de fechas, sin rellenar huecos.
    """
    raise NotImplementedError


def load_risk_free(data_dir: str = "data") -> pd.Series:
    """Tasa libre de riesgo diaria en decimal a partir de `^IRX` (÷ 100 ÷ 252)."""
    raise NotImplementedError


def audit_prices(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Auditoría de calidad de datos: huecos, fechas descartadas y coherencia OHLC."""
    raise NotImplementedError


def block_dates(config: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Fechas de inicio y fin de los bloques train, validation y test (SPEC punto 1)."""
    raise NotImplementedError
