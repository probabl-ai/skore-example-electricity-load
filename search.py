import json
import pickle

from sklearn.metrics import mean_absolute_percentage_error

import electricity_load_forecasting as elf

output_dir = elf.get_output_dir("search_")
env = elf.get_env()
pred = elf.make_data_op(horizon=24)

storage = f"sqlite:///{output_dir / 'optuna'}"
study_name = "randomized_search"
split = pred.skb.train_test_split(environment=env, split_func=elf.train_test_split)
search = pred.skb.make_randomized_search(
    backend="optuna",
    n_iter=64,
    scoring="neg_mean_absolute_percentage_error",
    n_jobs=1,
    refit=True,
    cv=elf.Splitter(),
    storage=storage,
    study_name=study_name,
)

search.fit(split["train"])
with open(output_dir / "search.pickle", "wb") as f:
    pickle.dump(search, f)

search.results_.to_csv(output_dir / "search_results.csv", index=False)
predictions = search.predict(split["test"])
results = split["X_test"].with_columns(
    true_load_mw=split["y_test"], predicted_load_mw=predictions
)
results.write_parquet(output_dir / "predictions.parquet")
test_score = mean_absolute_percentage_error(split["y_test"], predictions)
print(f"MAPE: {test_score}")
(output_dir / "score.json").write_text(json.dumps({"mape": test_score}), "utf-8")
fig = elf.plot_predictions(results)
fig.write_html(output_dir / "cv_predictions_plot.html")

search.plot_results().show(renderer="browser")
