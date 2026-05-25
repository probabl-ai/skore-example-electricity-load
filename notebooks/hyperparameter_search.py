# %% [markdown]
# # Hyperparameter search
#
# We load the dataop we dumped in the previous notebook and search for the best
# hyperparameters with optuna.

# %%
from pathlib import Path
import pickle

results_dir = Path(".") / "results"

# we can use 24 horizons instead
with open(results_dir / "learner_3_horizons.pickle", "rb") as f:
    pred = pickle.load(f).data_op


env = {"start": "2021-03-23", "end": "2025-05-31"}

# %% [markdown]
# We keep the last split of the default splitter as a held-out test set on
# which to validate the selected pipeline.

# %%
for outer_split in pred.skb.iter_cv_splits(env):
    pass

outer_split["X_test"]

# %% [markdown]
# We use persistent storage for our optuna database so we can resume or inspect
# it after the current process exits.

# %%
storage = f"sqlite:///{results_dir / 'optuna.sqlite'}"
print(f"Check search progress with:\noptuna-dashboard {storage}")
study_name = f"randomized_search"

search = pred.skb.make_randomized_search(
    backend="optuna",
    n_iter=64,
    n_jobs=1,
    refit="neg_mape_average",
    storage=storage,
    study_name=study_name,
)

search.fit(outer_split["train"])
with open(results_dir / "randomized_search.pickle", "wb") as f:
    pickle.dump(search, f)

with open(results_dir / "best_learner.pickle", "wb") as f:
    pickle.dump(search.best_learner_, f)

search.results_.to_csv("search_results.csv", index=False)
search.score(outer_split["test"])

# %%
search.plot_results()

# %%
# make an example prediction
search.predict({"start": "2025-06-27T15:00:00", "end": None})
