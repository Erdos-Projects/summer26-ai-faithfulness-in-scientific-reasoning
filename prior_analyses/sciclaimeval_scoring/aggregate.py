"""Per-cell mode/mean across passes; CSV export."""
import csv
from collections import Counter, defaultdict
from statistics import mean

def aggregate(conn, dims):
    cells = defaultdict(list)
    q = "SELECT item_id,dim_code,score FROM annotation WHERE parse_ok=1 AND dim_code IN ({})".format(
        ",".join("?" * len(dims)))
    for item_id, dim, score in conn.execute(q, dims):
        cells[(item_id, dim)].append(score)
    out = []
    for (item_id, dim), vals in sorted(cells.items()):
        out.append({"item_id": item_id, "dim_code": dim, "n_passes": len(vals),
                    "mode": Counter(vals).most_common(1)[0][0], "mean": mean(vals),
                    "min": min(vals), "max": max(vals)})
    return out

def export_csv(conn, dims, path):
    rows = aggregate(conn, dims)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item_id","dim_code","n_passes","mode","mean","min","max"])
        w.writeheader(); w.writerows(rows)
    return path
