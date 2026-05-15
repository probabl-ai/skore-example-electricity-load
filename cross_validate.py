import pickle
import json

import skore

import electricity_load_forecasting as elf

output_dir = elf.get_output_dir("cross_validate_")

env = elf.get_env()
pred = elf.make_data_op(horizons=(1, 12, 24))

# %%
# optional: make skrub reports
pred.skb.full_report(environment=env, output_dir=output_dir / "full_report")

split = pred.skb.train_test_split(environment=env, split_func=elf.train_test_split)
learner = pred.skb.make_learner()
learner.report(
    environment=split["train"],
    mode="fit",
    title="fit",
    output_dir=output_dir / "fit_report",
)
learner.report(
    environment=split["test"],
    mode="predict",
    title="predict",
    output_dir=output_dir / "predict_report",
)

# %%
cv_predictions, scores = elf.cross_val_predict(pred, env)
print(cv_predictions)
cv_predictions.write_parquet(output_dir / "cv_predictions.parquet")

fig = elf.plot_predictions(cv_predictions)
fig.write_html(output_dir / "cv_predictions_plot.html")
fig.show(renderer="browser")


fig = elf.plot_predictions(cv_predictions, horizons=(1,))
fig.write_html(output_dir / "cv_predictions_1h_plot.html")
fig.show(renderer="browser")

fig = elf.plot_predictions(cv_predictions, horizons=(24,))
fig.write_html(output_dir / "cv_predictions_24h_plot.html")
fig.show(renderer="browser")
