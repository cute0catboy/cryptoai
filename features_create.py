"""
Модуль для генерации технических и статистических признаков на основе часовых свечей.

Доступные признаки:
    - ret(): процентная доходность за N часов.
    - volatility(): скользящая волатильность по окну N часов.
    - vol_ratio(): отношение текущего объема к среднему за N часов.
    - price_to_sma(): относительное отклонение цены закрытия от SMA за N часов.

все докстринги написал gemini
"""

import numpy as np
import pandas as pd


class FeatureCreator:
    """Генератор признаков (фичей) на основе временного ряда OHLCV-данных."""

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Инициализирует генератор признаков.

        Args:
            data: DataFrame с историческими свечами. Обязательно должен содержать колонки 'close' (цена закрытия) и 'volume' (объем торгов).
        """
        self.data = data.copy()

    def ret(self, hours: int = 1) -> pd.Series:
        """
        Рассчитывает процентную доходность цены закрытия за указанный период.

        Формула: (Close_t / Close_{t - hours}) - 1

        Args:
            hours: Временной лаг (количество часов) для расчета доходности. По умолчанию 1 (часовая доходность).

        Returns:
            pd.Series: Временной ряд процентных доходностей.
        """
        return self.data["close"].pct_change(periods=hours)

    def volatility(self, hours: int = 24) -> pd.Series:
        """
        Рассчитывает скользящую волатильность (стандартное отклонение) за окно.

        Формула: std(ret_{t - hours + 1}, ..., ret_t), где ret — одночасовая доходность.

        Args:
            hours: Размер скользящего окна в часах. По умолчанию 24.

        Returns:
            pd.Series: Временной ряд выборочного стандартного отклонения (ddof=1).
        """
        hourly_ret = self.ret(hours=1)
        return hourly_ret.rolling(window=hours).std(ddof=1)

    def vol_ratio(self, hours: int = 24) -> pd.Series:
        """
        Рассчитывает коэффициент относительного объема торгов (Volume Ratio).

        Формула: Volume_t / Mean(Volume_{t - hours + 1}, ..., Volume_t)

        Args:
            hours: Размер скользящего окна в часах для усреднения объема. По умолчанию 24.

        Returns:
            pd.Series: Временной ряд отношения текущего объема к среднему.
        """
        rolling_mean_vol = self.data["volume"].rolling(window=hours).mean()
        return self.data["volume"] / rolling_mean_vol.replace(0, np.nan)

    def price_to_sma(self, hours: int = 24) -> pd.Series:
        """
        Рассчитывает относительное отклонение цены закрытия от простой скользящей средней.

        Формула: (Close_t / SMA_{hours}) - 1

        Args:
            hours: Период простой скользящей средней (SMA) в часах. По умолчанию 24.

        Returns:
            pd.Series: Временной ряд нормализованного отклонения цены от SMA.
        """
        sma = self.data["close"].rolling(window=hours).mean()
        return (self.data["close"] / sma) - 1
