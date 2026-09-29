# Data quality report

## Split overlap

- Train conversations: 128
- Validation conversations: 21
- Test conversations: 39
- intersection(train, val): 0
- intersection(train, test): 0
- intersection(val, test): 0
- Leakage detected: False

Missing IDs are **reported and never silently dropped**.
Repair rule: fill missing IDs only when they sit between two valid IDs of the same conversation and the turn gap equals the number of missing rows.
Unrepairable rows are written to `flagged_missing_ids.csv` and excluded from conversation reconstruction.

## train

- Rows: 8322
- Missing IDs: 66
- Malformed IDs: 100
- ID repairs applied: 0
- Missing utterances: 0
- Missing speakers: 0
- Missing emotion labels: 0
- Duplicate rows: 0

## val

- Rows: 1191
- Missing IDs: 0
- Malformed IDs: 0
- ID repairs applied: 0
- Missing utterances: 2
- Missing speakers: 0
- Missing emotion labels: 0
- Duplicate rows: 0

## test

- Rows: 2010
- Missing IDs: 0
- Malformed IDs: 0
- ID repairs applied: 0
- Missing utterances: 0
- Missing speakers: 0
- Missing emotion labels: 0
- Duplicate rows: 0
