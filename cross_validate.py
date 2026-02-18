import datetime
import json

import utils

output_dir = utils.get_output_dir("cross_validate_")
info_file = output_dir / "info.json"
info = {
    "commit": utils.last_commit_hash(),
    "date": datetime.datetime.now().isoformat(),
}
info_file.write_text(json.dumps(info), "utf-8")

env = utils.get_env()
pred = utils.make_data_op(horizon=24)
pred.skb.full_report(environment=env, output_dir=output_dir / "full_report")

split = pred.skb.train_test_split(environment=env, split_func=utils.train_test_split)
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

cv_predictions, cv_scores = utils.cross_val_predict(pred, environment=env)
cv_predictions.write_parquet(output_dir / "cv_predictions.parquet")

info["cv_results"] = cv_scores
info_file.write_text(json.dumps(info), "utf-8")

fig = utils.plot_predictions(cv_predictions)
fig.write_html(output_dir / "cv_predictions_plot.html")
fig.show(renderer="browser")
