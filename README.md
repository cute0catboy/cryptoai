# INTRODUCTION

this is the ai which predicts the stock exchange rate

right now: btc/usdt, 1h candles from binance (ccxt, last 1000 rows).
target: will the next hour close be higher than the current one (1 / 0).
only the data pipeline is done (features, split, clip, normalize, plots). no model yet.

# files:

<blockquote>
<pre><code>main.py            #the pipeline itself, run this
data_get.py        #get_data(): ohlcv from binance -> DataFrame with DatetimeIndex
features_create.py #FeatureCreator: ret, volatility, vol_ratio, price_to_sma
mpl_plots.py       #all plots (see below)
plots/             #created automatically, one png per plot
data/              #created automatically, empty for now</code></pre>
</blockquote>

run:

<blockquote>
<pre><code>pip install ccxt pandas numpy matplotlib
python main.py</code></pre>
</blockquote>

note: get_data() needs internet (public binance api, no key). the data is the last 1000 hours, so it is different every run.

# how to create pipeline with pandas:

data validation pipeline:

1. get the data. ccxt returns a list, make a df with DatetimeIndex (data_get.py):

<blockquote>
<pre><code>data = data_get.get_data()   #columns: open, high, low, close, volume
#if there are nan in ohlcv (not the case for binance), ffill them first:
s_ff = s.ffill()</code></pre>
</blockquote>

2. create y (1 if next day close is higher). use float, so the last row can be nan (it has no real target).

<blockquote>
<pre><code>next_close = data["close"].shift(-1)
y_raw = (next_close > data["close"]).astype(float)
y_raw[next_close.isna()] = np.nan</code></pre>
</blockquote>

3. prepare features you need. all windows are in hours (1 row = 1 hour).
features from FeatureCreator (features_create.py, only needs "close" and "volume"):

<blockquote>
<pre><code>features = fc.FeatureCreator(data)

ret4 = features.ret(4)                  #close.pct_change(4)
ret12 = features.ret(12)
ret24 = features.ret(24)
volatility6 = features.volatility(6)    #std of 1h returns in window (ddof=1)
volatility24 = features.volatility(24)
volatility72 = features.volatility(72)
vol_rat24 = features.vol_ratio(24)      #volume / rolling mean of volume
ptc24 = features.price_to_sma(24)       #close / SMA - 1</code></pre>
</blockquote>

3.1. candle and volume features (directly from OHLCV, no class):

<blockquote>
<pre><code>hl_range = (data["high"] - data["low"]) / data["close"]
body = (data["close"] - data["open"]) / data["open"]
vol_change = data["volume"].pct_change(1)</code></pre>
</blockquote>

3.2. some extra things (not used in main.py).
<blockquote>
<pre><code>data["dow"] = data.index.day_of_week
wm_df = pd.DataFrame({"ret": ret4, "dow": data["dow"]})
week_mean = wm_df.groupby("dow")["ret"].mean()</code></pre>
</blockquote>

all features (11): ret_4, ret_12, ret_24, hl_range, body, volatility_6, volatility_24, volatility_72, volume_change, volume_ratio_24, price_to_sma_24

4. build X and y from one table with one dropna(), so rows always match by date.
pct_change can give inf (volume 0 -> division by zero), so inf -> nan first.
the first row is set by the longest window (volatility_72 -> the day with index 72).

<blockquote>
<pre><code>data_new = (
    pd.DataFrame({
        "ret_4": ret4, "ret_12": ret12, "ret_24": ret24,
        "hl_range": hl_range, "body": body,
        "volatility_6": volatility6, "volatility_24": volatility24, "volatility_72": volatility72,
        "volume_change": vol_change, "volume_ratio_24": vol_rat24,
        "price_to_sma_24": ptc24,
        "y": y_raw,
    })
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
)
print(data_new.shape)
print(data_new.iloc[0])
X = data_new[["ret_4", "ret_12", "ret_24", "hl_range", "body",
              "volatility_6", "volatility_24", "volatility_72",
              "volume_change", "volume_ratio_24", "price_to_sma_24"]]
y = data_new["y"].astype(int)
assert not X.isna().any().any()</code></pre>
</blockquote>

4.1. resample your df if you need (not used now, data is already 1h).
<blockquote>
<pre><code>weekly = df.resample("W").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
})</code></pre>
</blockquote>


5. history split of X and y for train, validation, test (only by position, never shuffle):

<blockquote>
<pre><code>n_samples = X.shape[0]
split_70 = int(n_samples * 0.7)
split_85 = int(n_samples * 0.85)

X_train = X.iloc[:split_70]
y_train = y.iloc[:split_70]
X_validation = X.iloc[split_70:split_85]
y_validation = y.iloc[split_70:split_85]
X_test = X.iloc[split_85:]
y_test = y.iloc[split_85:]</code></pre>
</blockquote>

5.1. check class balance (share of y=1 in every split). about 50% is normal, big difference = look at plot_class_balance / plot_target_over_time.

<blockquote>
<pre><code>print(y_train.mean(), y_validation.mean(), y_test.mean())</code></pre>
</blockquote>

6. assert X_... with y_... - must be with similar shapes:

<blockquote>
<pre><code>assert X_train.shape[0] == y_train.shape[0]
assert X_validation.shape[0] == y_validation.shape[0]
assert X_test.shape[0] == y_test.shape[0]</code></pre>
</blockquote>

7. assert X and y restored (pd.concat keeps dates, .equals compares values and index):

<blockquote>
<pre><code>assert pd.concat([X_train, X_validation, X_test]).equals(X)
assert pd.concat([y_train, y_validation, y_test]).equals(y)</code></pre>
</blockquote>

8. clip X_train, X_validation, X_test with bounds from train only, per column (3 sigma).
axis=1 here means "align low/high with columns".
Note: pandas std() uses ddof=1, numpy ddof=0.

<blockquote>
<pre><code>mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
clip_low = mu - 3 * sd
clip_high = mu + 3 * sd

X_train_clipped = X_train.clip(lower=clip_low, upper=clip_high, axis=1)
X_validation_clipped = X_validation.clip(lower=clip_low, upper=clip_high, axis=1)
X_test_clipped = X_test.clip(lower=clip_low, upper=clip_high, axis=1)</code></pre>
</blockquote>

8.1. diagnostics table for train (main.py prints it): unique values, inf count, bounds and % of clipped values in train / val / test.
about 1-3% clipped in train is normal. much more in val / test = the distribution has shifted (see plot_clip_share).

<blockquote>
<pre><code>diag_df = pd.DataFrame({
    "Unique_vals": X_train.nunique(),
    "Inf_count": np.isinf(X_train).sum(),
    "Clip_low": clip_low,
    "Clip_high": clip_high,
    "Train_clipped_cnt": (X_train != X_train_clipped).sum(),
    "Train_clipped_%": ((X_train != X_train_clipped).mean() * 100).round(1),
    "Val_clipped_%": ((X_validation != X_validation_clipped).mean() * 100).round(1),
    "Test_clipped_%": ((X_test != X_test_clipped).mean() * 100).round(1),
})
print(diag_df.to_string())</code></pre>
</blockquote>

9. minmax normalize X (min and max only from clipped train):

<blockquote>
<pre><code>X_min = X_train_clipped.min(axis=0)
X_max = X_train_clipped.max(axis=0)

mm = X_max - X_min
mm[mm == 0] = 1.0

X_train_normalized = (X_train_clipped - X_min) / mm
X_validation_normalized = (X_validation_clipped - X_min) / mm
X_test_normalized = (X_test_clipped - X_min) / mm</code></pre>
</blockquote>

10. final assert (train must be exactly [0, 1], val and test may go outside):

<blockquote>
<pre><code>assert np.allclose(X_train_normalized.min(), 0)
assert np.allclose(X_train_normalized.max(), 1)

for name, arr in [("train", X_train_normalized),
                  ("val", X_validation_normalized),
                  ("test", X_test_normalized)]:
    print(name, arr.min(axis=0).to_list(), arr.max(axis=0).to_list())
    assert not arr.isna().any().any()</code></pre>
</blockquote>

# features_create.py:
class FeatureCreator(data), data is df with "close" and "volume". every method returns pd.Series with the same index as data (first rows are nan because of the window).
docstrings are by gemini.

<blockquote>
<pre><code>ret(hours=1)          #close.pct_change(hours)
volatility(hours=24)  #rolling std (ddof=1) of 1h returns
vol_ratio(hours=24)   #volume / rolling mean of volume (mean 0 -> nan)
price_to_sma(hours=24) #close / SMA - 1</code></pre>
</blockquote>

# mpl_plots.py:
Its the art of vibecoding. (dangerous to edit: it was supposed to be small, then claude got out of control and wrote 700 lines)

every function: draws, saves to plots/name.png, shows (if show=True), closes the figure, returns one number or None (count of values outside bounds etc.).
every function has the same last params: name, show, plots_dir.

here are defs (return in brackets):

price / time
<blockquote>
<pre><code>plot_price_splits         #close + SMA + train/val/test ranges (None)
plot_price_volume         #price and volume, 3 : 1 (None)
plot_target_over_time     #rolling share of y=1 + splits (None)</code></pre>
</blockquote>

target
<blockquote>
<pre><code>plot_class_balance        #share of y=1 in train/val/test (float: max |share - 0.5|)
plot_target_correlation   #correlation of each feature with y (float: max |corr|)
plot_target_by_quantile   #mean of y per quantile of one feature (float: spread of y)
plot_acf                  #autocorrelation of a series, 95% band (int: lags outside band)</code></pre>
</blockquote>

features
<blockquote>
<pre><code>plot_correlation_matrix   #feature x feature heatmap (int: pairs with |corr| >= threshold)
plot_hist_grid            #histograms of all features + clip bounds (int: values outside bounds)</code></pre>
</blockquote>

clipping / normalization
<blockquote>
<pre><code>plot_hist_bounds          #one feature + clip bounds (int: values outside)
plot_before_after         #one feature before / after clip (int: values changed by clip)
plot_clip_share           #% of clipped values per feature and split (float: max %)
plot_normalized           #one normalized feature, 3 splits in time (int: val + test outside [0, 1])
plot_normalized_box       #boxplots of all normalized features, 3 splits (int: val + test outside [0, 1])
plot_split_distributions  #one feature: train / val / test distributions (float: KS distance train vs test)</code></pre>
</blockquote>

example:

<blockquote>
<pre><code>import mpl_plots as mpl

mpl.plot_price_splits(data["close"], X_train, X_validation, X_test)
mpl.plot_correlation_matrix(X_train)        #train only, so test does not influence feature choice
mpl.plot_hist_grid(X_train, clip_low, clip_high)
n = mpl.plot_hist_bounds(X_train["ret_4"], clip_low["ret_4"], clip_high["ret_4"])
ks = mpl.plot_split_distributions(X_train_normalized["ret_4"], X_validation_normalized["ret_4"], X_test_normalized["ret_4"])
#ks above ~0.2 = noticeable shift between train and test</code></pre>
</blockquote>

other info in docstring

# graphics with matplotlib(unnecessary, i hate this, better use mpl_plots.py):

Rules:
- always use `fig, ax`, not `plt.plot`
- savefig BEFORE plt.show()
- one file name per plot
- same bins and same scale when comparing two plots

0. setup (once, at the top of the file):

<blockquote>
<pre><code>import matplotlib.pyplot as plt

PLOTS = Path(__file__).parent.resolve() / "plots"
PLOTS.mkdir(exist_ok=True)</code></pre>
</blockquote>

1. base template (every plot starts like this):

<blockquote>
<pre><code>fig, ax = plt.subplots(figsize=(10, 4))

ax.set_title("title")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()                      # only if there are label=...
fig.tight_layout()               # no cut labels
fig.savefig(PLOTS / "name.png", dpi=120)
plt.show()
plt.close(fig)                   # free memory</code></pre>
</blockquote>

2. line (Series with DatetimeIndex, dates go to x axis automatically):

<blockquote>
<pre><code>ax.plot(df["close"], color="#661A1A", label="close")
ax.plot(df["close"].rolling(20).mean(), color="#141A6E", label="SMA20")</code></pre>
</blockquote>

3. histogram (distribution of values):

<blockquote>
<pre><code>ax.hist(X_train["ret_4"], bins=30, color="#4C72B0", edgecolor="white")</code></pre>
</blockquote>

4. vertical and horizontal lines (clip bounds, 0 and 1 for normalized data):

<blockquote>
<pre><code>ax.axvline(clip_low["ret_4"], color="red", linestyle="--", label="clip low")
ax.axvline(clip_high["ret_4"], color="red", linestyle="--", label="clip high")
ax.axhline(0, color="black", linestyle="--")
ax.axhline(1, color="black", linestyle="--")</code></pre>
</blockquote>

5. colored date ranges (train / validation / test):

<blockquote>
<pre><code>ax.axvspan(X_train.index[0], X_train.index[-1], color="#996A0561", label="train")
ax.axvspan(X_validation.index[0], X_validation.index[-1], color="#1F3E8061", label="validation")
ax.axvspan(X_test.index[0], X_test.index[-1], color="#1F805F61", label="test")</code></pre>
</blockquote>

6. several series on one axis (train, validation, test have different dates, so lines go one after another):

<blockquote>
<pre><code>ax.plot(X_train_normalized["ret_4"], label="train")
ax.plot(X_validation_normalized["ret_4"], label="validation")
ax.plot(X_test_normalized["ret_4"], label="test")</code></pre>
</blockquote>

7. grid 1x2, before and after (axes is an array, sharey and common bins make plots comparable):

<blockquote>
<pre><code>col = "volume_change"
bins = np.linspace(X_train[col].min(), X_train[col].max(), 31)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
axes[0].hist(X_train[col], bins=bins, color="#4C72B0", edgecolor="white")
axes[0].set_title("before clip")
axes[1].hist(X_train_clipped[col], bins=bins, color="#55A868", edgecolor="white")
axes[1].set_title("after clip")
for ax in axes:
    ax.set_xlabel(col)
axes[0].set_ylabel("count")
fig.tight_layout()
fig.savefig(PLOTS / "before_after_clip.png", dpi=120)</code></pre>
</blockquote>

8. counting values outside of bounds (no loops, no "&lt;" in code):

<blockquote>
<pre><code># outside clip bounds (train)
outside = (~X_train["ret_4"].between(clip_low["ret_4"], clip_high["ret_4"])).sum()

for name, arr in [("val", X_validation_normalized), ("test", X_test_normalized)]:
    print(name, (~arr["ret_4"].between(0, 1)).sum())

print((X_train != X_train_clipped).sum())</code></pre>
</blockquote>
