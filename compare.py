import pandas as pd
import skrub
import skore

project = skore.Project("jerome-workspace-1/electricity_forecasting", mode="local")
summary = project.summarize().frame()
print(summary)
skrub.TableReport(
    summary, n_rows=100000, plot_distributions=False, compute_associations=False
).open()
ids = summary.index.get_level_values(1).tolist()
keys = summary["key"].tolist()
comparison = skore.compare({key: project.get(i) for key, i in zip(keys, ids)})
print('ok')
metric_summary = comparison.metrics.summarize().frame()
print(metric_summary)
skrub.TableReport(
    metric_summary, n_rows=100000, plot_distributions=False, compute_associations=False
).open()
