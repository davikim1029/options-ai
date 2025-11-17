import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

class OptionPermutationLoader:
    """
    Streams data from option_permutations in consistent DataFrame batches,
    converting permutations back into grouped sequences so the training code
    sees the same structure as when reading CSV snapshots.
    """

    def __init__(self, db_path: str, batch_size: int = 50_000):
        self.db_path = Path(db_path)
        self.batch_size = batch_size

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    @staticmethod
    def _compute_sequence(lifetime_rows):
        """Convert a list of permutation rows → chronological snapshot sequence."""
        seq = []
        # reconstruct sequence using buy snapshot fields
        for row in lifetime_rows:
            snap = {
                "timestamp": row["buy_timestamp"],
                "strikePrice": row["strikePrice"],
                "optionType": row["optionType"],
                "moneyness": row["moneyness"],
                "bid": row["bid"],
                "ask": row["ask"],
                "delta": row["delta"],
                "gamma": row["gamma"],
                "theta": row["theta"],
                "vega": row["vega"],
                "rho": row["rho"],
                "iv": row["iv"],
                "lastPrice": row["buy_price"],
                "spread": row["spread"],
                "midPrice": row["midPrice"],
                "volume": row.get("volume", 0),
                "openInterest": row.get("openInterest", 0),
                "inTheMoney": row.get("inTheMoney", 0),
                "bidSize": row.get("bidSize", 0),
                "askSize": row.get("askSize", 0),
                "nearPrice": row.get("nearPrice", 0),
                "daysToExpiration": row["daysToExpiration"],
            }
            seq.append(snap)

        # sort by timestamp
        seq = sorted(seq, key=lambda x: x["timestamp"])
        return seq

    @staticmethod
    def _compute_targets(perms):
        """Compute the final return and hold days from permutations."""
        # permutations already sorted by buy then sell
        first = perms[0]
        last = perms[-1]

        buy_price = first["buy_price"]
        final_price = last["sell_price"]

        if buy_price is None or buy_price == 0:
            ret = 0.0
        else:
            ret = (final_price - buy_price) / buy_price

        # compute days difference
        t0 = datetime.fromisoformat(first["buy_timestamp"])
        t1 = datetime.fromisoformat(last["sell_timestamp"])
        days = (t1 - t0).days
        return float(ret), int(days)

    # ----------------------------------------------------
    # Streaming generator
    # ----------------------------------------------------
    def stream(self):
        """
        Yields DataFrame batches shaped like the CSV used in your streaming trainer:
        - sequence (list of dicts)
        - predicted_return
        - predicted_hold_days
        - feature columns (1-snapshot fields)
        """

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        # Fetch each OSI key (one lifetime at a time)
        cur.execute("SELECT DISTINCT osiKey FROM option_permutations")
        osi_keys = [row[0] for row in cur.fetchall()]

        batch_rows = []

        for osi in osi_keys:
            cur.execute("""
                SELECT *
                FROM option_permutations
                WHERE osiKey = ?
                ORDER BY buy_timestamp ASC, sell_timestamp ASC
            """, (osi,))
            rows = [dict(r) for r in cur.fetchall()]
            if not rows:
                continue

            # Rebuild the lifetime snapshot sequence
            sequence = self._compute_sequence(rows)

            # Targets
            pred_return, pred_days = self._compute_targets(rows)

            # Pick the FIRST snapshot as static features
            first = sequence[0]

            batch_rows.append({
                "sequence": sequence,
                "predicted_return": pred_return,
                "predicted_hold_days": pred_days,
                # Include flattened feature columns identical to your CSV
                **first
            })

            if len(batch_rows) >= self.batch_size:
                yield pd.DataFrame(batch_rows)
                batch_rows = []

        # Final batch
        if batch_rows:
            yield pd.DataFrame(batch_rows)

        conn.close()
