import electricity_load_forecasting as elf


pred = elf.make_data_op(horizon=24)
g = pred.skb.draw_graph()
with open('graph.svg', 'wb') as f:
    f.write(g.svg)
with open('graph.png', 'wb') as f:
    f.write(g.png)
