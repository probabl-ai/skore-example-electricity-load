import pandas as pd
import skore

project = skore.Project("jerome-workspace-1/electricity_forecasting", mode="local")
summary = pd.DataFrame(project.summarize())
ids = summary.index.get_level_values(1).tolist()
keys = summary["key"].tolist()
comparison = skore.compare({key: project.get(i) for key, i in zip(keys, ids)})
print(comparison.metrics.summarize().frame())
