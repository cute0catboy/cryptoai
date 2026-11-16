'''
mpl_plots - plots for the data pipeline.

rules:
- always fig, ax (never plt.plot)
- savefig BEFORE plt.show()
- one file name per plot
- same bins and same scale when two plots are compared

every function:
- draws, saves to PLOTS / name, shows (if show=True), closes the figure
- returns one number or None (count of values outside bounds, etc.)

contents:
  price / time
    plot_price_splits          close + SMA + train/val/test ranges
    plot_price_volume          price and volume (3 : 1)
    plot_target_over_time      rolling share of y=1 over time + splits
  target
    plot_class_balance         share of y=1 in train / val / test
    plot_target_correlation    correlation of each feature with y
    plot_target_by_quantile    mean of y per quantile of one feature
    plot_acf                   autocorrelation of a series (e.g. ret_1)
  features
    plot_correlation_matrix    feature x feature correlation heatmap
    plot_hist_grid             histograms of all features + clip bounds
  clipping / normalization
    plot_hist_bounds           one feature + clip bounds
    plot_before_after          one feature before / after clip
    plot_clip_share            % of clipped values per feature and split
    plot_normalized            one normalized feature, 3 splits in time
    plot_normalized_box        boxplots of all normalized features, 3 splits
    plot_split_distributions   one feature: train / val / test distributions
'''
#в этот модуль опасно лезть, тк я не люблю матплотлиб я это все навайбкодил но в какой то момент клод вышел из под контроля и написал 700 строк

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    PLOTS = Path(__file__).parent.resolve() / "plots"
except NameError:  # notebook / interactive session: no __file__
    PLOTS = Path.cwd() / "plots"

C_TRAIN = "#996A05"
C_VAL = "#1F3E80"
C_TEST = "#1F805F"
C_CLOSE = "#661A1A"
C_SMA = "#141A6E"


def _finish(fig, name: str, show: bool, plots_dir: Path) -> None:
    '''
    tight_layout -> savefig -> show -> close. order matters:
    savefig before show, otherwise the file can be empty.
    '''
    plots_dir.mkdir(parents=True, exist_ok=True)  # FIX: parents=True
    fig.tight_layout()
    fig.savefig(plots_dir / name, dpi=120)
    if show:
        plt.show()
    plt.close(fig)


def _span_splits(ax, X_train, X_validation, X_test, labels: bool = True) -> None:
    '''
    color train / validation / test ranges on a time axis (index = dates).
    '''
    for X, color, lbl in ((X_train, C_TRAIN, "train"),
                          (X_validation, C_VAL, "validation"),
                          (X_test, C_TEST, "test")):
        ax.axvspan(X.index[0], X.index[-1], color=color, alpha=0.2,
                   label=lbl if labels else None)


def _ks_stat(a: pd.Series, b: pd.Series) -> float:
    '''
    Kolmogorov-Smirnov distance between two samples (max gap between
    empirical CDFs), 0 = same distribution, 1 = completely different.
    NaN are ignored. Pure numpy, no scipy.
    '''
    a = np.sort(a.dropna().to_numpy())
    b = np.sort(b.dropna().to_numpy())
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


# =====================================================================
# price / time
# =====================================================================

def plot_price_splits(close: pd.Series,
                      X_train: pd.DataFrame,
                      X_validation: pd.DataFrame,
                      X_test: pd.DataFrame,
                      window: int = 20,
                      name: str = "price_splits.png",
                      show: bool = True,
                      plots_dir: Path = PLOTS) -> None:
    '''
    close + SMA(window) with colored train / validation / test ranges.

    params:
    close: pd.Series with DatetimeIndex
    X_train, X_validation, X_test: DataFrames, only their index (dates) is used

    return:
    None
    '''
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(close, color=C_CLOSE, lw=1.2, label="close")
    ax.plot(close.rolling(window).mean(), color=C_SMA, lw=1.5, label=f"SMA{window}")

    ax.axvspan(X_train.index[0], X_train.index[-1], color=C_TRAIN, alpha=0.2, label="train")
    ax.axvspan(X_validation.index[0], X_validation.index[-1], color=C_VAL, alpha=0.2, label="validation")
    ax.axvspan(X_test.index[0], X_test.index[-1], color=C_TEST, alpha=0.2, label="test")

    ax.set_title(f"Close and SMA{window}")
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)


def plot_price_volume(df: pd.DataFrame,
                      name: str = "price_volume.png",
                      show: bool = True,
                      plots_dir: Path = PLOTS) -> None:
    '''
    price on top, volume below, common x axis, heights 3 : 1.
    bar width is taken from the data step (median distance between
    timestamps), so it works for hourly and daily data.

    params:
    df: DataFrame with DatetimeIndex and columns "close", "volume"

    return:
    None
    '''
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(df["close"], color=C_CLOSE, label="close")
    ax1.set_ylabel("price")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # FIX: width=1.0 was "1 day" - bars on hourly data merged into a wall.
    # matplotlib date units are days, so step / 1 day gives the right width.
    step = df.index.to_series().diff().median()
    width = (step / pd.Timedelta(days=1)) * 0.9 if pd.notna(step) else 1.0
    ax2.bar(df.index, df["volume"], width=width, color="gray")
    ax2.set_ylabel("volume")

    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)


def plot_target_over_time(y: pd.Series,
                          X_train: pd.DataFrame,
                          X_validation: pd.DataFrame,
                          X_test: pd.DataFrame,
                          window: int = 72,
                          name: str = "target_over_time.png",
                          show: bool = True,
                          plots_dir: Path = PLOTS) -> None:
    '''
    rolling share of y=1 (price went up next hour) over time, with the
    train / validation / test ranges. dashed line = 0.5 (coin flip).
    shows whether the class balance drifts between splits.

    params:
    y: pd.Series of 0/1 with DatetimeIndex
    X_train, X_validation, X_test: only their index is used
    window: rolling window in rows (72 = 3 days of hourly data)

    return:
    None
    '''
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y.rolling(window).mean(), color="black", lw=1.2,
            label=f"share of y=1 (rolling {window})")
    ax.axhline(0.5, color="red", linestyle="--", label="0.5")
    _span_splits(ax, X_train, X_validation, X_test)

    ax.set_title("Target balance over time")
    ax.set_xlabel("date")
    ax.set_ylabel("share of y=1")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)


# =====================================================================
# target
# =====================================================================

def plot_class_balance(y_train: pd.Series,
                       y_validation: pd.Series,
                       y_test: pd.Series,
                       name: str = "class_balance.png",
                       show: bool = True,
                       plots_dir: Path = PLOTS) -> float:
    '''
    bars: share of y=1 in train / validation / test, dashed line at 0.5.
    values are written above the bars.

    params:
    y_train, y_validation, y_test: pd.Series of 0/1

    return:
    float - max |share - 0.5| over the three splits
    (0 = perfectly balanced, 0.05 = one split is 55% / 45%)
    '''
    shares = [float(y_train.mean()), float(y_validation.mean()), float(y_test.mean())]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["train", "validation", "test"], shares,
                  color=[C_TRAIN, C_VAL, C_TEST])
    ax.axhline(0.5, color="red", linestyle="--", label="0.5")
    for b, s in zip(bars, shares):
        ax.text(b.get_x() + b.get_width() / 2, s + 0.01, f"{s:.1%}",
                ha="center", va="bottom")

    ax.set_title("Share of y=1 (price up)")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1)
    ax.legend()
    _finish(fig, name, show, plots_dir)

    return float(max(abs(s - 0.5) for s in shares))


def plot_target_correlation(X: pd.DataFrame,
                            y: pd.Series,
                            method: str = "pearson",
                            name: str = "target_correlation.png",
                            show: bool = True,
                            plots_dir: Path = PLOTS) -> float:
    '''
    horizontal bars: correlation of every feature with the target y,
    sorted. for hourly crypto data all bars are expected to be small
    (|r| < 0.05); a big bar is a reason to check for leakage.

    params:
    X: DataFrame of features (use train only to avoid peeking)
    y: pd.Series, same index as X
    method: "pearson" or "spearman"

    return:
    float - max |correlation| between a feature and y
    '''
    corr = X.corrwith(y, method=method).sort_values()

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(corr) + 1.5))
    colors = ["#C44E52" if v < 0 else "#4C72B0" for v in corr]
    ax.barh(corr.index, corr.to_numpy(), color=colors)
    ax.axvline(0, color="black", lw=0.8)

    ax.set_title(f"Correlation with target ({method})")
    ax.set_xlabel("correlation")
    ax.grid(True, axis="x", alpha=0.3)
    _finish(fig, name, show, plots_dir)

    return float(corr.abs().max())


def plot_target_by_quantile(feature: pd.Series,
                            y: pd.Series,
                            q: int = 10,
                            name: str = "target_by_quantile.png",
                            show: bool = True,
                            plots_dir: Path = PLOTS) -> float:
    '''
    split one feature into q equal-count groups (deciles by default) and
    show the mean of y in every group. dashed line = overall mean of y.
    catches non-linear dependencies that correlation does not show:
    a flat picture = the feature carries no signal.

    params:
    feature: pd.Series (for example X_train["ret_1"])
    y: pd.Series of 0/1, same index
    q: number of groups

    return:
    float - max group mean minus min group mean (spread of y across groups)
    '''
    df = pd.concat([feature, y], axis=1, keys=["f", "y"]).dropna()
    groups = pd.qcut(df["f"], q=q, duplicates="drop")
    means = df["y"].groupby(groups, observed=True).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(means) + 1), means.to_numpy(), color="#4C72B0", edgecolor="white")
    ax.axhline(df["y"].mean(), color="red", linestyle="--", label="overall mean")

    ax.set_title(f"Mean of y by quantile of {feature.name}")
    ax.set_xlabel(f"quantile group of {feature.name} (1 = lowest)")
    ax.set_ylabel("mean of y")
    ax.set_xticks(range(1, len(means) + 1))
    ax.set_ylim(0, 1)
    ax.legend()
    _finish(fig, name, show, plots_dir)

    return float(means.max() - means.min())


def plot_acf(series: pd.Series,
             lags: int = 48,
             name: str = "acf.png",
             show: bool = True,
             plots_dir: Path = PLOTS) -> int:
    '''
    autocorrelation of a series for lags 1..lags with approximate 95%
    bounds (+-1.96 / sqrt(n)). typical use: ret_1. bars outside the band
    mean the past of the series says something about its future.
    (about 5% of lags outside the band is expected by pure chance.)

    params:
    series: pd.Series (for example X_train["ret_1"])
    lags: max lag in rows

    return:
    int - how many lags are outside the 95% band
    '''
    x = series.dropna().to_numpy(dtype=float)
    n = len(x)
    lags = min(lags, n - 1)
    x = x - x.mean()
    denom = float(np.sum(x ** 2))
    acf = np.array([
        np.sum(x[:-k] * x[k:]) / denom if denom > 0 else np.nan
        for k in range(1, lags + 1)
    ])
    bound = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.vlines(range(1, lags + 1), 0, acf, color="#4C72B0", lw=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(bound, color="red", linestyle="--", label="95% band")
    ax.axhline(-bound, color="red", linestyle="--")

    ax.set_title(f"Autocorrelation: {series.name}")
    ax.set_xlabel("lag (rows)")
    ax.set_ylabel("acf")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finish(fig, name, show, plots_dir)

    return int(np.sum(np.abs(acf) > bound))


# =====================================================================
# features
# =====================================================================

def plot_correlation_matrix(X: pd.DataFrame,
                            method: str = "pearson",
                            threshold: float = 0.8,
                            annot: bool = True,
                            name: str = "corr_matrix.png",
                            show: bool = True,
                            plots_dir: Path = PLOTS) -> int:
    '''
    feature x feature correlation matrix as a heatmap.
    fixed scale -1..1 and a diverging colormap (red = positive,
    blue = negative, white = 0), so different matrices are comparable.
    highly correlated pairs (ret_1 vs body, volatility_24 vs
    volatility_72 ...) are redundant features.
    use train only, so the test period does not influence feature choice.

    params:
    X: DataFrame of features (X_train or X_train_clipped)
    method: "pearson" (linear) or "spearman" (rank, robust to outliers)
    threshold: pairs with |corr| >= threshold are counted
    annot: write the numbers inside the cells

    return:
    int - number of feature pairs (each pair once, without the diagonal)
    with |corr| >= threshold
    '''
    corr = X.corr(method=method)
    n = corr.shape[0]
    values = corr.to_numpy()

    size = max(6.0, 0.75 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index)

    if annot:
        for i in range(n):
            for j in range(n):
                v = values[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.6 else "black")

    fig.colorbar(im, ax=ax, label=f"{method} correlation", fraction=0.046, pad=0.04)
    ax.set_title(f"Feature correlation matrix ({method})")
    _finish(fig, name, show, plots_dir)

    upper = np.triu_indices(n, k=1)
    return int(np.sum(np.abs(values[upper]) >= threshold))  # NaN -> False


def plot_hist_grid(X: pd.DataFrame,
                   clip_low: pd.Series | None = None,
                   clip_high: pd.Series | None = None,
                   bins: int = 30,
                   ncols: int = 4,
                   name: str = "hist_grid.png",
                   show: bool = True,
                   plots_dir: Path = PLOTS) -> int:
    '''
    histograms of ALL features in one picture (one small plot per
    feature). if clip_low / clip_high are given, red dashed lines show
    the clip bounds. every feature has its own x scale (they are not
    compared with each other). good first look at skew and outliers.

    params:
    X: DataFrame (train only)
    clip_low, clip_high: pd.Series indexed by feature names (optional)
    ncols: plots per row

    return:
    int - total number of values outside the clip bounds
    (0 if bounds are not given). NaN are ignored.
    '''
    cols = list(X.columns)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()

    outside = 0
    for ax, col in zip(axes_flat, cols):
        s = X[col].dropna()
        ax.hist(s, bins=bins, color="#4C72B0", edgecolor="white")
        if clip_low is not None and clip_high is not None:
            ax.axvline(clip_low[col], color="red", linestyle="--", lw=1)
            ax.axvline(clip_high[col], color="red", linestyle="--", lw=1)
            outside += int((~s.between(clip_low[col], clip_high[col])).sum())
        ax.set_title(col, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes_flat[len(cols):]:  # hide empty cells
        ax.set_visible(False)

    fig.suptitle("Feature distributions (train)")
    _finish(fig, name, show, plots_dir)

    return outside


# =====================================================================
# clipping / normalization
# =====================================================================

def plot_hist_bounds(values: pd.Series,
                     low: float,
                     high: float,
                     bins: int = 30,
                     name: str = "hist_bounds.png",
                     show: bool = True,
                     plots_dir: Path = PLOTS) -> int:
    '''
    histogram of values with vertical lines at low and high (clip bounds).
    NaN are ignored (neither drawn nor counted).

    params:
    values: pd.Series (train only)
    low, high: numbers, for example clip_low["ret"], clip_high["ret"]

    return:
    int - how many values are strictly outside [low, high]
    '''
    clean = values.dropna()  # FIX: NaN used to be counted as "outside"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(clean, bins=bins, color="#4C72B0", edgecolor="white")
    ax.axvline(low, color="red", linestyle="--", label="clip low")
    ax.axvline(high, color="red", linestyle="--", label="clip high")

    ax.set_title(f"Train: {values.name}")
    ax.set_xlabel(str(values.name))
    ax.set_ylabel("count")
    ax.legend()
    _finish(fig, name, show, plots_dir)

    return int((~clean.between(low, high)).sum())


def plot_before_after(before: pd.Series,
                      after: pd.Series,
                      bins: int = 30,
                      name: str = "before_after_clip.png",
                      show: bool = True,
                      plots_dir: Path = PLOTS) -> int:
    '''
    two histograms side by side: before and after clip.
    bins are made once from `before` and used for both;
    sharex=True and sharey=True, so the plots are comparable.

    params:
    before: pd.Series (for example X_train["vol_change"])
    after: pd.Series (for example X_train_clipped["vol_change"]), same index

    return:
    int - how many values clip changed (NaN in both = not changed)
    '''
    b = before.dropna()
    a = after.dropna()
    lo, hi = b.min(), b.max()
    if lo == hi:  # FIX: constant column gave zero-width bins -> error
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)

    # FIX: sharex=True added (same scale on x too)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True, sharey=True)
    axes[0].hist(b, bins=edges, color="#4C72B0", edgecolor="white")
    axes[0].set_title("before clip")
    axes[1].hist(a, bins=edges, color="#55A868", edgecolor="white")
    axes[1].set_title("after clip")
    for ax in axes:
        ax.set_xlabel(str(before.name))
    axes[0].set_ylabel("count")
    _finish(fig, name, show, plots_dir)

    # FIX: NaN != NaN is True, so NaN used to be counted as "changed"
    both_nan = before.isna() & after.isna()
    return int(((before != after) & ~both_nan).sum())


def plot_clip_share(X_train: pd.DataFrame,
                    X_validation: pd.DataFrame,
                    X_test: pd.DataFrame,
                    clip_low: pd.Series,
                    clip_high: pd.Series,
                    name: str = "clip_share.png",
                    show: bool = True,
                    plots_dir: Path = PLOTS) -> float:
    '''
    grouped bars: % of values outside the clip bounds for every feature,
    separately for train / validation / test (same as the Train/Val/Test
    clipped % columns of the diagnostics table, but visual).
    for 3 sigma on train about 1-3% is normal. much more on val / test
    means the distribution has shifted (regime change).

    params:
    X_train, X_validation, X_test: DataFrames BEFORE clipping
    clip_low, clip_high: pd.Series indexed by feature names (train stats)

    return:
    float - max % of clipped values over all features and splits
    '''
    def share(X: pd.DataFrame) -> pd.Series:
        out = X.lt(clip_low, axis=1) | X.gt(clip_high, axis=1)
        return out.mean() * 100  # NaN compare as False

    tr, va, te = share(X_train), share(X_validation), share(X_test)
    x = np.arange(len(tr))
    w = 0.27

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w, tr.to_numpy(), w, color=C_TRAIN, label="train")
    ax.bar(x, va.to_numpy(), w, color=C_VAL, label="validation")
    ax.bar(x + w, te.to_numpy(), w, color=C_TEST, label="test")

    ax.set_xticks(x)
    ax.set_xticklabels(tr.index, rotation=45, ha="right")
    ax.set_title("Share of clipped values per feature")
    ax.set_ylabel("% of rows")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _finish(fig, name, show, plots_dir)

    return float(max(tr.max(), va.max(), te.max()))


def plot_normalized(train: pd.Series,
                    validation: pd.Series,
                    test: pd.Series,
                    name: str = "normalized.png",
                    show: bool = True,
                    plots_dir: Path = PLOTS) -> int:
    '''
    one normalized column for train / validation / test on one axis.
    dashed lines at 0 and 1 = train range (min and max come from train only).

    params:
    train, validation, test: pd.Series with dates in index

    return:
    int - how many values of validation + test are outside [0, 1]
    (NaN are ignored). one number, as the module rules require.
    '''
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(train, color=C_TRAIN, label="train")
    ax.plot(validation, color=C_VAL, label="validation")
    ax.plot(test, color=C_TEST, label="test")
    ax.axhline(0, color="black", linestyle="--")
    ax.axhline(1, color="black", linestyle="--")

    # FIX: title used the file name ("Normalized normalized.png"), now column name
    col = train.name if train.name is not None else "value"
    ax.set_title(f"Normalized: {col}")
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.legend()
    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)

    # FIX: NaN used to be counted as outside; dict return replaced by one int
    n_val = int((~validation.dropna().between(0, 1)).sum())
    n_test = int((~test.dropna().between(0, 1)).sum())
    return n_val + n_test


def plot_normalized_box(X_train_normalized: pd.DataFrame,
                        X_validation_normalized: pd.DataFrame,
                        X_test_normalized: pd.DataFrame,
                        name: str = "normalized_box.png",
                        show: bool = True,
                        plots_dir: Path = PLOTS) -> int:
    '''
    boxplots of ALL normalized features: three boxes per feature
    (train / validation / test), one common y scale. dashed lines at 0
    and 1 = train range. shows in one picture which features are
    shifted or stretched on val / test relative to train.

    params:
    X_train_normalized, X_validation_normalized, X_test_normalized:
    DataFrames with the same columns, after min-max normalization

    return:
    int - how many values of validation + test are outside [0, 1]
    (NaN are ignored)
    '''
    cols = list(X_train_normalized.columns)
    n = len(cols)
    sets = ((X_train_normalized, C_TRAIN, "train"),
            (X_validation_normalized, C_VAL, "validation"),
            (X_test_normalized, C_TEST, "test"))

    fig, ax = plt.subplots(figsize=(max(9.0, 0.9 * n + 2), 5))
    handles = []
    for k, (df, color, lbl) in enumerate(sets):
        data = [df[c].dropna().to_numpy() for c in cols]
        pos = np.arange(n) * 4 + k
        bp = ax.boxplot(data, positions=pos, widths=0.8, patch_artist=True,
                        showfliers=False,
                        medianprops={"color": "black"})
        for box in bp["boxes"]:
            box.set_facecolor(color)
            box.set_alpha(0.6)
        handles.append(bp["boxes"][0])

    ax.axhline(0, color="black", linestyle="--")
    ax.axhline(1, color="black", linestyle="--")
    ax.set_xticks(np.arange(n) * 4 + 1)
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_title("Normalized features: train / validation / test")
    ax.set_ylabel("normalized value")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles, ["train", "validation", "test"])
    _finish(fig, name, show, plots_dir)

    out = 0
    for df in (X_validation_normalized, X_test_normalized):
        out += int((~df.stack().between(0, 1)).sum())  # stack() drops NaN
    return out


def plot_split_distributions(train: pd.Series,
                             validation: pd.Series,
                             test: pd.Series,
                             bins: int = 30,
                             name: str = "split_distribution.png",
                             show: bool = True,
                             plots_dir: Path = PLOTS) -> float:
    '''
    one feature, three distributions (train / validation / test) as
    step histograms on the SAME bins and the same axes. density=True,
    because the splits have different sizes. if the lines differ a lot,
    the feature drifts in time and the model trained on train will not
    work on test.

    params:
    train, validation, test: pd.Series of one feature (raw, clipped or
    normalized - the same version for all three)

    return:
    float - Kolmogorov-Smirnov distance train vs test (0..1,
    0 = same distribution; above ~0.2 is a noticeable shift)
    '''
    parts = [s.dropna() for s in (train, validation, test)]
    lo = min(p.min() for p in parts)
    hi = max(p.max() for p in parts)
    if lo == hi:  # constant column: avoid zero-width bins
        lo, hi = lo - 0.5, hi + 0.5
    edges = np.linspace(lo, hi, bins + 1)  # one set of bins for all three

    fig, ax = plt.subplots(figsize=(8, 4))
    for p, color, lbl in zip(parts, (C_TRAIN, C_VAL, C_TEST),
                             ("train", "validation", "test")):
        ax.hist(p, bins=edges, density=True, histtype="step", lw=1.8,
                color=color, label=lbl)

    col = train.name if train.name is not None else "value"
    ax.set_title(f"Distribution by split: {col}")
    ax.set_xlabel(str(col))
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finish(fig, name, show, plots_dir)

    return _ks_stat(train, test)
