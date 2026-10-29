import argparse

import skrub
import skore

parser = argparse.ArgumentParser()
parser.add_argument("-m", default=None)
args = parser.parse_args()

project = skore.Project("electricity_forecasting", mode="local")
summary = project.summarize().frame()
print(summary)
r = skrub.TableReport(
    summary, n_rows=100000, plot_distributions=False, compute_associations=False
).open()
ids = summary.index.get_level_values(1).tolist()
keys = summary["key"].tolist()
key_to_id = dict(zip(keys, ids))
if args.m is not None:
    key_to_id = {k: v for k, v in key_to_id.items() if k in args.m.split()}
comparison = skore.compare(
    {key: project.get(i) for key, i in key_to_id.items()}
)
metric_summary = comparison.metrics.summarize().frame()
print(metric_summary)
skrub.TableReport(
    metric_summary, n_rows=100000, plot_distributions=False, compute_associations=False
).open()
