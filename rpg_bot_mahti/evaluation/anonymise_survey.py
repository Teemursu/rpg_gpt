"""Strip identifying metadata from a raw Qualtrics export of the
"Dialogue Engagement & Relevance" survey (added 2026 during repository
consolidation; not part of the original thesis code).

The raw exports (kept outside git, see ../../data/README.md) contain IP
addresses, geolocation, timestamps, response IDs and a free-text field.
Only the "games played" multiple-choice answer and the Likert items
(game_* / ai_* columns) are kept, which is all the analysis scripts use.

Usage:
    python anonymise_survey.py RAW_EXPORT.csv OUT.csv
"""
import sys

import pandas as pd

raw, out = sys.argv[1], sys.argv[2]
sep = ";" if open(raw, encoding="utf-8-sig").readline().count(";") > 5 else ","
df = pd.read_csv(raw, sep=sep, encoding="utf-8-sig")
keep = ["games_played"] + [c for c in df.columns if c.startswith(("game_", "ai_"))]
df = df[keep].sample(frac=1.0, random_state=0).reset_index(drop=True)  # drop time order
df.to_csv(out, sep=";", index=False)
print(f"wrote {len(df)} rows x {len(keep)} columns to {out}")
