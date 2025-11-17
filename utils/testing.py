from logger.logger_singleton import getLogger
import sqlite3
from constants import DB_PATH as db_path


def copy_all_snapshot_data():
    logger = getLogger()
    # Current file directory

    insert_sql = """
    INSERT INTO option_lifetimes (
        osiKey, timestamp, symbol, optionType, strikePrice, lastPrice,
        bid, ask, bidSize, askSize, volume, openInterest, nearPrice,
        inTheMoney, delta, gamma, theta, vega, rho, iv,
        daysToExpiration, spread, midPrice, moneyness, processed
    )
    SELECT
        osiKey, timestamp, symbol, optionType, strikePrice, lastPrice,
        bid, ask, bidSize, askSize, volume, openInterest, nearPrice,
        inTheMoney, delta, gamma, theta, vega, rho, iv,
        daysToExpiration, spread, midPrice, moneyness, 0
    FROM option_snapshots;
    """
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Count rows before insert
        cur.execute("SELECT COUNT(*) FROM option_lifetimes;")
        before_count = cur.fetchone()[0]

        # Execute the insert
        cur.execute(insert_sql)
        conn.commit()

        # Count rows after insert
        cur.execute("SELECT COUNT(*) FROM option_lifetimes;")
        after_count = cur.fetchone()[0]

        rows_inserted = after_count - before_count
        logger.logMessage(f"Insert successful! {rows_inserted} rows added.")

    except sqlite3.Error as e:
        logger.logMessage(f"SQLite error: {e}")
    finally:
        if conn:
            conn.close()
