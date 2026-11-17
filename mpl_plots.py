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
  model evaluation (y vs y_hat)
    plot_match_over_time       match / mismatch of y and y_hat in time
    plot_confusion_matrix      confusion matrix (counts and % per true class)
    plot_metrics_by_split      accuracy train / val / test vs baseline
    plot_proba_hist            predicted probabilities by true class
    plot_roc_curve             ROC curve and AUC
    plot_calibration           predicted probability vs real frequency
    plot_threshold_scan        accuracy / precision / recall / F1 vs threshold
    plot_strategy_equity       equity curve of a trading rule vs buy & hold
    plot_training_history      loss (and metric) per epoch
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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


# =====================================================================
# model evaluation (y vs y_hat)
#
# y      - true target (0 / 1), pd.Series with DatetimeIndex
# y_hat  - model output: labels 0 / 1 OR probabilities of class 1.
#          a label is made as (y_hat >= threshold), default 0.5.
#          numpy arrays (model.predict) are accepted: they get y.index.
# y_proba - probabilities only (functions that need a score, not a label)
#
# important: choose the threshold on VALIDATION, report on TEST once.
# =====================================================================

C_MATCH = "#2CA02C"
C_MISS = "#D62728"


def _pair(y, y_hat) -> tuple[pd.Series, pd.Series]:
    '''
    align y and y_hat by index, drop NaN. y_hat may be a numpy array
    (then it takes the index of y). returns (y as int, y_hat as float).
    '''
    if not isinstance(y_hat, pd.Series):
        y_hat = pd.Series(np.asarray(y_hat, dtype=float).ravel(), index=y.index)
    df = pd.concat([y, y_hat], axis=1, keys=["y", "y_hat"]).dropna()
    return df["y"].astype(int), df["y_hat"].astype(float)


def _metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    '''
    accuracy, precision, recall, f1 for class 1. 0.0 instead of 0 / 0.
    '''
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def plot_match_over_time(y: pd.Series,
                         y_hat,
                         threshold: float = 0.5,
                         window: int = 24,
                         name: str = "match_over_time.png",
                         show: bool = True,
                         plots_dir: Path = PLOTS) -> float:
    '''
    match / mismatch of y and y_hat in time.
    top: rolling accuracy (window rows), dashed red = 0.5 (coin flip),
    dotted = overall accuracy.
    bottom: one thin line per row, green = model was right,
    red = model was wrong. shows WHEN the model works: long red
    stretches = regime where it fails.

    params:
    y: pd.Series of 0/1 with DatetimeIndex
    y_hat: labels 0/1 or probabilities (pd.Series or numpy array)
    threshold: y_hat >= threshold -> class 1
    window: rolling window in rows (24 = one day of hourly data)

    return:
    float - overall accuracy (share of matches)
    '''
    yy, p = _pair(y, y_hat)
    match = (yy == (p >= threshold).astype(int))
    acc = float(match.mean())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot(match.astype(float).rolling(window).mean(), color="black", lw=1.2,
             label=f"rolling accuracy ({window})")
    ax1.axhline(0.5, color="red", linestyle="--", label="0.5")
    ax1.axhline(acc, color=C_VAL, linestyle=":", label=f"overall {acc:.1%}")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("accuracy")
    ax1.set_title("Match of y and y_hat")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    colors = np.where(match.to_numpy(), C_MATCH, C_MISS)
    ax2.vlines(match.index, 0, 1, colors=colors, lw=1)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("date")
    ax2.legend(handles=[Line2D([0], [0], color=C_MATCH, lw=3, label="match"),
                        Line2D([0], [0], color=C_MISS, lw=3, label="mismatch")],
               loc="upper right", ncol=2, fontsize=8)

    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)
    return acc


def plot_confusion_matrix(y: pd.Series,
                          y_hat,
                          threshold: float = 0.5,
                          name: str = "confusion_matrix.png",
                          show: bool = True,
                          plots_dir: Path = PLOTS) -> float:
    '''
    confusion matrix 2 x 2: rows = true class, columns = predicted class.
    every cell: count and % of its true-class row. color = % of the row
    (fixed scale 0..1). diagonal = matches. if the model always says
    "up", one column is empty - this plot shows that at once, while
    plain accuracy hides it. precision / recall / F1 are in the title.

    params:
    y: pd.Series of 0/1
    y_hat: labels 0/1 or probabilities
    threshold: y_hat >= threshold -> class 1

    return:
    float - accuracy
    '''
    yy, p = _pair(y, y_hat)
    yv = yy.to_numpy()
    pred = (p >= threshold).astype(int).to_numpy()

    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for q in (0, 1):
            cm[t, q] = int(np.sum((yv == t) & (pred == q)))
    row = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row, out=np.zeros(cm.shape, dtype=float), where=row > 0)
    acc, prec, rec, f1 = _metrics(yv, pred)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.1%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["down (0)", "up (1)"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["down (0)", "up (1)"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"acc {acc:.3f} | prec {prec:.3f} | rec {rec:.3f} | F1 {f1:.3f}",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="share of true class", fraction=0.046, pad=0.04)
    _finish(fig, name, show, plots_dir)
    return float(acc)


def plot_metrics_by_split(y_train: pd.Series, y_hat_train,
                          y_validation: pd.Series, y_hat_validation,
                          y_test: pd.Series, y_hat_test,
                          threshold: float = 0.5,
                          name: str = "metrics_by_split.png",
                          show: bool = True,
                          plots_dir: Path = PLOTS) -> float:
    '''
    accuracy of the model on train / validation / test (colored bars,
    black whiskers = 95% binomial confidence interval) next to a
    baseline "always predict the majority class of train" (gray).
    dashed line = 0.5. the model is useful only if its bar is above
    the baseline AND the whisker does not touch it. big gap between
    train and validation = overfitting. with ~150 rows in a split the
    interval is about +-8 p.p., so small differences are noise.

    params:
    y_*: pd.Series of 0/1 for each split
    y_hat_*: labels or probabilities for each split
    threshold: y_hat >= threshold -> class 1

    return:
    float - accuracy(train) minus accuracy(test) (overfitting gap)
    '''
    pairs = [_pair(a, b) for a, b in ((y_train, y_hat_train),
                                      (y_validation, y_hat_validation),
                                      (y_test, y_hat_test))]
    majority = int(pairs[0][0].mean() >= 0.5)  # from train only

    accs, errs, bases = [], [], []
    for yy, p in pairs:
        acc = float(np.mean(yy.to_numpy() == (p >= threshold).astype(int).to_numpy()))
        n = len(yy)
        accs.append(acc)
        errs.append(1.96 * np.sqrt(acc * (1 - acc) / n))
        bases.append(float(np.mean(yy.to_numpy() == majority)))

    x = np.arange(3)
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, accs, w, yerr=errs, capsize=4,
           color=[C_TRAIN, C_VAL, C_TEST], label="model")
    ax.bar(x + w / 2, bases, w, color="gray", alpha=0.6,
           label=f"baseline (always {majority})")
    for xi, a in zip(x, accs):
        ax.text(xi - w / 2, 0.02, f"{a:.1%}", ha="center", color="white",
                fontweight="bold")
    ax.axhline(0.5, color="red", linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(["train", "validation", "test"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by split")
    ax.legend(loc="upper right")
    _finish(fig, name, show, plots_dir)
    return float(accs[0] - accs[2])


def plot_proba_hist(y: pd.Series,
                    y_proba,
                    threshold: float = 0.5,
                    bins: int = 20,
                    name: str = "proba_hist.png",
                    show: bool = True,
                    plots_dir: Path = PLOTS) -> float:
    '''
    histogram of predicted probabilities, separately for rows where the
    truth is "up" and "down", on the same bins (0..1) and the same axes.
    if the two histograms lie on top of each other, the model does not
    separate the classes. if everything sits in a narrow peak near 0.5,
    the model is unsure about everything. dashed line = threshold.

    params:
    y: pd.Series of 0/1
    y_proba: probabilities of class 1 (0..1)

    return:
    float - mean probability for y=1 minus mean probability for y=0
    (0 = no separation)
    '''
    yy, p = _pair(y, y_proba)
    edges = np.linspace(0, 1, bins + 1)
    p0 = p[yy == 0]
    p1 = p[yy == 1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(p0, bins=edges, alpha=0.6, color=C_MISS, label="true: down (0)")
    ax.hist(p1, bins=edges, alpha=0.6, color=C_MATCH, label="true: up (1)")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold {threshold}")

    ax.set_title("Predicted probability by true class")
    ax.set_xlabel("predicted probability of up")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)
    ax.legend()
    _finish(fig, name, show, plots_dir)

    if len(p0) == 0 or len(p1) == 0:
        return float("nan")
    return float(p1.mean() - p0.mean())


def plot_roc_curve(y: pd.Series,
                   y_proba,
                   name: str = "roc_curve.png",
                   show: bool = True,
                   plots_dir: Path = PLOTS) -> float:
    '''
    ROC curve (false positive rate vs true positive rate) over all
    thresholds. diagonal = random guessing (AUC 0.5). AUC does not
    depend on the threshold, so it is a good way to compare models.
    for hourly crypto an AUC of 0.50-0.53 is normal; above 0.6 on
    test - check for leakage first.

    params:
    y: pd.Series of 0/1
    y_proba: probabilities (or any score, higher = more "up")

    return:
    float - AUC (area under the curve), nan if only one class in y
    '''
    yy, s = _pair(y, y_proba)
    yv = yy.to_numpy()
    sv = s.to_numpy()

    order = np.argsort(-sv, kind="mergesort")
    ys, ss = yv[order], sv[order]
    idx = np.r_[np.where(np.diff(ss))[0], len(ys) - 1]  # last row of each score
    tps = np.cumsum(ys)[idx]
    fps = (idx + 1) - tps
    pos, neg = tps[-1], fps[-1]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    if pos == 0 or neg == 0:
        auc = float("nan")
    else:
        tpr = np.r_[0.0, tps / pos]
        fpr = np.r_[0.0, fps / neg]
        auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        ax.plot(fpr, tpr, color=C_VAL, lw=2, label=f"model (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="red", linestyle="--", label="random (0.5)")

    ax.set_title("ROC curve")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    _finish(fig, name, show, plots_dir)
    return auc


def plot_calibration(y: pd.Series,
                     y_proba,
                     bins: int = 10,
                     name: str = "calibration.png",
                     show: bool = True,
                     plots_dir: Path = PLOTS) -> float:
    '''
    calibration: predicted probability (x) vs how often "up" really
    happened in that bucket (y). diagonal = perfectly calibrated
    ("70% means 7 of 10"). below the plot - how many predictions fall
    into each bucket (the same x axis): points built on a few rows are
    not reliable.

    params:
    y: pd.Series of 0/1
    y_proba: probabilities (0..1)
    bins: number of equal-width probability buckets

    return:
    float - ECE (expected calibration error): weighted mean
    |real frequency - predicted probability|. 0 = ideal.
    '''
    yy, p = _pair(y, y_proba)
    yv = yy.to_numpy()
    pv = p.to_numpy()
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(pv, edges[1:-1]), 0, bins - 1)

    xs, ys, counts = [], [], []
    ece = 0.0
    for b in range(bins):
        m = which == b
        if m.sum() == 0:
            continue
        xs.append(pv[m].mean())
        ys.append(yv[m].mean())
        counts.append(int(m.sum()))
        ece += m.sum() / len(pv) * abs(yv[m].mean() - pv[m].mean())

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax1.plot([0, 1], [0, 1], color="red", linestyle="--", label="ideal")
    ax1.plot(xs, ys, color=C_VAL, marker="o", label=f"model (ECE {ece:.3f})")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("real share of up")
    ax1.set_title("Calibration")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.hist(pv, bins=edges, color="gray", edgecolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("predicted probability of up")
    ax2.set_ylabel("count")
    _finish(fig, name, show, plots_dir)
    return float(ece)


def plot_threshold_scan(y: pd.Series,
                        y_proba,
                        name: str = "threshold_scan.png",
                        show: bool = True,
                        plots_dir: Path = PLOTS) -> float:
    '''
    accuracy, precision, recall and F1 for every threshold from 0.05
    to 0.95. vertical dotted line = threshold with the best accuracy.
    use VALIDATION data here and then fix the threshold for test:
    choosing it on test is a leak.

    params:
    y: pd.Series of 0/1
    y_proba: probabilities (0..1)

    return:
    float - threshold with the best accuracy
    '''
    yy, p = _pair(y, y_proba)
    yv = yy.to_numpy()
    pv = p.to_numpy()
    grid = np.linspace(0.05, 0.95, 91)
    res = np.array([_metrics(yv, (pv >= t).astype(int)) for t in grid])  # (T, 4)
    best = float(grid[int(np.nanargmax(res[:, 0]))])

    fig, ax = plt.subplots(figsize=(8, 4))
    for k, lbl in enumerate(("accuracy", "precision", "recall", "F1")):
        ax.plot(grid, res[:, k], lw=1.8 if k == 0 else 1.2, label=lbl)
    ax.axhline(0.5, color="red", linestyle="--", lw=0.8)
    ax.axvline(best, color="black", linestyle=":", label=f"best acc @ {best:.2f}")

    ax.set_title("Metrics vs threshold")
    ax.set_xlabel("threshold")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _finish(fig, name, show, plots_dir)
    return best


def plot_strategy_equity(next_ret: pd.Series,
                         y_hat,
                         threshold: float = 0.5,
                         long_short: bool = True,
                         fee: float = 0.001,
                         name: str = "strategy_equity.png",
                         show: bool = True,
                         plots_dir: Path = PLOTS) -> float:
    '''
    equity curve of the rule "y_hat = 1 -> buy, y_hat = 0 -> sell
    (or stay out)" against buy & hold. accuracy can be 53% and still
    lose money after fees, so this is the real test of the model.
    start = 1.0. the fee is charged on every change of position.

    params:
    next_ret: pd.Series, price return from this row to the NEXT row,
    for example close.shift(-1) / close - 1. it is the future, so it is
    used only here, never as a feature. last row (NaN) is dropped.
    y_hat: labels or probabilities, same index
    threshold: y_hat >= threshold -> class 1
    long_short: True = +1 / -1 position, False = +1 / 0 (long only)
    fee: cost per unit of position change (0.001 = 0.1%)

    return:
    float - total return of the strategy (0.05 = +5%)
    '''
    if not isinstance(y_hat, pd.Series):
        y_hat = pd.Series(np.asarray(y_hat, dtype=float).ravel(), index=next_ret.index)
    df = pd.concat([next_ret, y_hat], axis=1, keys=["r", "p"]).dropna()
    r = df["r"].astype(float)  # not _pair(): it would cast returns to int
    pred = (df["p"] >= threshold).astype(int)

    pos = np.where(pred.to_numpy() == 1, 1.0, -1.0 if long_short else 0.0)
    turnover = np.abs(np.diff(np.r_[0.0, pos]))
    strat = pos * r.to_numpy() - fee * turnover
    equity = pd.Series(np.cumprod(1 + strat), index=r.index)
    hold = pd.Series(np.cumprod(1 + r.to_numpy()), index=r.index)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hold, color="gray", lw=1.2, label=f"buy & hold ({hold.iloc[-1] - 1:+.1%})")
    ax.plot(equity, color=C_VAL, lw=1.6,
            label=f"strategy ({equity.iloc[-1] - 1:+.1%}, fee {fee:.2%})")
    ax.axhline(1.0, color="black", linestyle="--", lw=0.8)

    ax.set_title("Equity: " + ("long / short" if long_short else "long only"))
    ax.set_xlabel("date")
    ax.set_ylabel("equity (start = 1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    _finish(fig, name, show, plots_dir)
    return float(equity.iloc[-1] - 1)


def plot_training_history(train_loss,
                          val_loss,
                          train_metric=None,
                          val_metric=None,
                          metric_name: str = "accuracy",
                          name: str = "training_history.png",
                          show: bool = True,
                          plots_dir: Path = PLOTS) -> int:
    '''
    learning curves per epoch: loss on train and validation (left),
    and optionally a metric (right). dotted vertical line = epoch with
    the lowest validation loss. validation loss going up while train
    loss goes down = overfitting, stop earlier (early stopping).

    params:
    train_loss, val_loss: lists / arrays, one value per epoch
    train_metric, val_metric: optional lists, for example accuracy
    metric_name: label of the metric

    return:
    int - best epoch (1-based) by validation loss
    '''
    tl = np.asarray(train_loss, dtype=float)
    vl = np.asarray(val_loss, dtype=float)
    epochs = np.arange(1, len(tl) + 1)
    best = int(np.nanargmin(vl)) + 1

    with_metric = train_metric is not None and val_metric is not None
    ncols = 2 if with_metric else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols + 1, 4), squeeze=False)

    ax = axes[0, 0]
    ax.plot(epochs, tl, color=C_TRAIN, label="train")
    ax.plot(epochs, vl, color=C_VAL, label="validation")
    ax.axvline(best, color="black", linestyle=":", label=f"best epoch {best}")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if with_metric:
        ax = axes[0, 1]
        ax.plot(epochs, np.asarray(train_metric, dtype=float), color=C_TRAIN, label="train")
        ax.plot(epochs, np.asarray(val_metric, dtype=float), color=C_VAL, label="validation")
        ax.axvline(best, color="black", linestyle=":")
        ax.set_title(metric_name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.legend()

    _finish(fig, name, show, plots_dir)
    return best