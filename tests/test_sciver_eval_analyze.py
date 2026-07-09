import math
import sqlite3

from sciver_eval import analyze_paired as ap


def test_dprime_criterion_known_values():
    # hit=0.69, fa=0.11  ->  d' = z(.69) - z(.11);  c = -0.5*(z(.69)+z(.11))
    d, c = ap.dprime_criterion(0.69, 0.11)
    assert math.isclose(d, 1.7227, abs_tol=1e-3)
    assert math.isclose(c, 0.3655, abs_tol=1e-3)


def test_dprime_clamps_perfect_rates():
    # hit=1.0, fa=0.0 must not blow up to infinity
    d, c = ap.dprime_criterion(1.0, 0.0, n_signal=100, n_noise=100)
    assert math.isfinite(d) and d > 0


def test_mcnemar_symmetric_is_zero():
    stat, _ = ap.mcnemar(10, 10)
    assert math.isclose(stat, 0.0, abs_tol=1e-9)


def test_mcnemar_continuity_correction():
    # max(0,|b-c|-1)^2 / (b+c) = (|20-10|-1)^2 / 30 = 81/30 = 2.7
    stat, _ = ap.mcnemar(20, 10)
    assert math.isclose(stat, 2.7, abs_tol=1e-9)


def _seed(conn):
    conn.executescript("""
      CREATE TABLE prediction(run INTEGER, item_id TEXT, model TEXT, trial INTEGER,
                              predicted INTEGER, correct INTEGER, parse_ok INTEGER);
    """)
    rows = [
        # entailed run (2): item A correct, item B wrong
        (2, "A", "m", 1, 1, 1, 1), (2, "B", "m", 1, 0, 0, 1),
        # refuted run (3): item A correct, item B correct
        (3, "A", "m", 1, 0, 1, 1), (3, "B", "m", 1, 0, 1, 1),
    ]
    conn.executemany("INSERT INTO prediction VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()


def test_paired_metrics_driver():
    conn = sqlite3.connect(":memory:")
    _seed(conn)
    m = ap.paired_metrics(conn, run_ent=2, run_ref=3, model="m")
    assert m["n_items"] == 2
    assert math.isclose(m["acc_entailed"], 0.5, abs_tol=1e-9)   # A right, B wrong
    assert math.isclose(m["acc_refuted"], 1.0, abs_tol=1e-9)    # both right
    assert math.isclose(m["gap"], -0.5, abs_tol=1e-9)           # ent - ref
    # discordant pairs: B is refuted-only-correct -> c=1, b=0
    assert m["mcnemar_b"] == 0 and m["mcnemar_c"] == 1
