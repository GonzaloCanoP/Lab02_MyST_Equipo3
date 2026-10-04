"""Carga, descarga y auditoría de datos del portafolio (P1).

`download_prices` es la única función del proyecto que usa red (CLAUDE.md, sección 3).
"""

from pathlib import Path

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
# Tolerancia relativa de la coherencia OHLC en `audit_prices` (redondeo del ajuste de Yahoo).
OHLC_TOLERANCE = 1e-6


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


def _read_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV de `data/` con índice de fechas y columnas OHLCV en minúsculas."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {sorted(missing)}")
    return df[OHLCV_COLUMNS].sort_index()


def load_prices(tickers: list[str], data_dir: str = "data") -> dict[str, pd.DataFrame]:
    """Lee los CSV de `data/` y devuelve un DataFrame OHLCV por ticker con índice común.

    CLAUDE.md, sección 3: el índice común es la intersección de fechas de todos los tickers, sin
    rellenar huecos. Las fechas que un ticker tenía y se descartaron por la intersección quedan en
    `df.attrs["dropped_dates"]` para que `audit_prices` las reporte.

    Parameters
    ----------
    tickers : list[str]
        Tickers del portafolio, sin `^`.
    data_dir : str
        Carpeta con un `<TICKER>.csv` por activo.

    Returns
    -------
    dict[str, pd.DataFrame]
        Un DataFrame `open, high, low, close, volume` por ticker, todos con el mismo índice.
    """
    raw = {t: _read_csv(Path(data_dir) / _ticker_filename(t)) for t in tickers}
    common = raw[tickers[0]].index
    for df in raw.values():
        common = common.intersection(df.index)

    prices = {}
    for ticker, df in raw.items():
        aligned = df.loc[common].copy()
        aligned.attrs["dropped_dates"] = list(df.index.difference(common))
        prices[ticker] = aligned
    return prices


def load_risk_free(data_dir: str = "data") -> pd.Series:
    """Tasa libre de riesgo diaria en decimal a partir de `^IRX`.

    `^IRX` cotiza el rendimiento anual del T-Bill a 13 semanas en porcentaje; la tasa diaria es
    close ÷ 100 ÷ 252 (P1, "Cómo encaja").

    Returns
    -------
    pd.Series
        Tasa diaria por fecha, con nombre "rf".
    """
    irx = _read_csv(Path(data_dir) / _ticker_filename("^IRX"))
    return (irx["close"] / 100 / 252).rename("rf")


def audit_prices(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Auditoría de calidad de datos, un renglón por ticker. No corrige nada: solo reporta.

    CLAUDE.md, sección 3: todo hueco se reporta. La coherencia OHLC se evalúa con tolerancia
    relativa `OHLC_TOLERANCE`, porque el ajuste por dividendos de Yahoo produce diferencias de redondeo
    (~1e-8) entre `low`/`high` y `open`/`close` que no son errores de datos.

    Returns
    -------
    pd.DataFrame
        Columnas: n_obs, start, end, nan, non_positive, high_lt_low, open_outside, close_outside,
        zero_volume, dropped_dates.
    """
    rows = {}
    for ticker, df in prices.items():
        low = df["low"] * (1 - OHLC_TOLERANCE)
        high = df["high"] * (1 + OHLC_TOLERANCE)
        rows[ticker] = {
            "n_obs": len(df),
            "start": df.index.min(),
            "end": df.index.max(),
            "nan": int(df.isna().any(axis=1).sum()),
            "non_positive": int((df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()),
            "high_lt_low": int((df["high"] < df["low"]).sum()),
            "open_outside": int(((df["open"] < low) | (df["open"] > high)).sum()),
            "close_outside": int(((df["close"] < low) | (df["close"] > high)).sum()),
            "zero_volume": int((df["volume"] == 0).sum()),
            "dropped_dates": len(df.attrs.get("dropped_dates", [])),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def block_dates(config: dict) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """Fechas de inicio y fin, inclusivas, de los bloques train, validation y test (SPEC punto 1)."""
    return {
        name: (pd.Timestamp(start), pd.Timestamp(end))
        for name, (start, end) in config["blocks"].items()
    }
