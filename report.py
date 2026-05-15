import electricity_load_forecasting as elf

env = elf.get_env()
pred = elf.make_data_op()
pred.skb.eval(env)
pred.skb.full_report(env)
