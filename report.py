import electricity_load_forecasting as elf

env = elf.get_env()
pred = elf.make_data_op()
pred.skb.eval(env)
pred.skb.full_report(env)

learner = pred.skb.make_learner().fit(env)
learner.report(environment={"start": elf.get_new_date()}, mode="predict")
