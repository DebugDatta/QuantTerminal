"""Trade log and summary generation."""

import pandas as pd
from typing import List, Dict


def trade_log(trades: List[Dict], df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw trade list into a structured DataFrame."""
    if not trades:
        return pd.DataFrame()

    records = []
    for t in trades:
        idx = t["idx"]
        date = df.index[idx] if idx < len(df.index) else None
        records.append({
            "Date": date,
            "Action": t["action"].capitalize(),
            "Price": t.get("price", 0),
            "Shares": t.get("shares", 0),
            "Cost": t.get("cost", 0),
        })
    return pd.DataFrame(records)


def trade_summary(trades: List[Dict]) -> Dict:
    """Compute summary statistics from trade list."""
    if not trades:
        return {"total_trades": 0, "buys": 0, "sells": 0, "total_cost": 0}

    buys = len([t for t in trades if t["action"] in ("buy", "short")])
    sells = len([t for t in trades if t["action"] in ("sell", "cover")])
    total_cost = sum(t.get("cost", 0) for t in trades)

    return {
        "total_trades": len(trades),
        "buys": buys,
        "sells": sells,
        "total_cost": total_cost,
    }
