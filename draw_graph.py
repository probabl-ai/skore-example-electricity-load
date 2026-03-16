import electricity_load_forecasting as elf


pred = elf.make_data_op(horizon=24)
with open('graph.svg', 'wb') as f:
    f.write(pred.skb.draw_graph().svg)
