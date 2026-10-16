import pprint
import electricity_load_forecasting as elf

import skrub._data_ops._utils as u
u.has_graphviz = lambda: False   # pretend graphviz is missing

env = elf.get_env()
# quantile_strategy = "multiple_regressors"
quantile_strategy = None
pred = elf.make_data_op(quantile_strategy=quantile_strategy)
pred.skb.report(env)

# split = pred.skb.train_test_split(env)
# learner = pred.skb.make_learner().fit(split["train"])
# score = learner.score(split["test"])
# pprint.pprint(score)

# learner.report(environment={"start": elf.get_new_date()}, mode="predict")
# if quantile_strategy == "binning":
#     learner.report(
#         environment={
#             "start": elf.get_new_date(),
#             "quantiles": (0.01, 0.1, 0.5, 0.9, 0.99),
#         },
#         mode="predict",
#     )
