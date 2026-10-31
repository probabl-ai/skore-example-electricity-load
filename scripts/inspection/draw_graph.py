from pathlib import Path

import electricity_load_forecasting as elf

ROOT = Path(__file__).resolve().parents[2]

output_dir = ROOT / "results" / "inspection"
output_dir.mkdir(parents=True, exist_ok=True)
pred = elf.make_data_op(model="tabicl")
graph = pred.skb.draw_graph()
with open(output_dir / "graph.svg", "wb") as f:
    f.write(graph.svg)
with open(output_dir / "graph.png", "wb") as f:
    f.write(graph.png)
