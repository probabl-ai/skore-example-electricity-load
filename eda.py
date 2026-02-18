import polars as pl
import skrub

from utils import data_dir

# %%
for csv in sorted(data_dir().glob("*.csv")):
    df = pl.read_csv(csv, null_values=["N/A", "-"])
    skrub.TableReport(df, title=csv.name).open()

# %%
from utils import fetch_electricity_load

electricity_load = fetch_electricity_load().sort('time')
skrub.TableReport(electricity_load, title="all load data").open()

# %%
electricity_load.drop_nulls()['time'].max()

# %%
for (y,), df in electricity_load.group_by(pl.col('time').dt.year()):
    skrub.TableReport(df, title=str(y)).open()

# %% [markdown]
# We note that data stops in June 2025 but there are also a few null values
# before. Also towards the end of 2024 data started being sampled every 15
# minutes; before it was every hour
