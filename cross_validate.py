import json

import electricity_load_forecasting as elf

output_dir = elf.get_output_dir("cross_validate_")

env = elf.get_env()
pred = elf.make_data_op(horizons=(1, 12, 24))

# %%
# optional: make skrub reports
pred.skb.full_report(environment=env, output_dir=output_dir / "full_report")
split = pred.skb.train_test_split(environment=env)
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
cv_predictions.write_parquet(output_dir / "cv_predictions.parquet")
(output_dir / "scores.json").write_text(json.dumps(scores), "utf-8")

fig = elf.plot_predictions(cv_predictions)
fig.write_html(output_dir / "cv_predictions_plot.html")
fig.show(renderer="browser")


fig = elf.plot_predictions(cv_predictions, horizons=(1,))
fig.write_html(output_dir / "cv_predictions_1h_plot.html")
fig.show(renderer="browser")

fig = elf.plot_predictions(cv_predictions, horizons=(24,))
fig.write_html(output_dir / "cv_predictions_24h_plot.html")
fig.show(renderer="browser")

# %%
index = """
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>cross-validation result</title>
    </head>
    <body>
        <ul>
            <li><a href="full_report/index.html">data op report</a></li>
            <li><a href="fit_report/index.html">data op fit report</a></li>
            <li><a href="predict_report/index.html">data op predict report</a></li>
            <li><a href="cv_predictions_plot.html">multiple horizon predictions</a></li>
            <li><a href="cv_predictions_1h_plot.html">1-h horizon predictions</a></li>
            <li><a href="cv_predictions_24h_plot.html">24-h horizon predictions</a></li>
        </ul>
    </body>
</html>
"""
(output_dir / "index.html").write_text(index, "utf-8")
