# Quant Research MVP: reproducible factor experiments

This is a research demonstration, not trading advice or evidence of live returns. The first milestone evaluates **one fixed universe at daily frequency**. A candidate is a signal function that ranks the assets; the system either rejects it or retains it with a complete experiment record. Positive returns are not a completion criterion.

## Data and timing contract

- Input is a user-supplied CSV with `date,symbol,close` columns, plus a manifest recording its source, redistribution permission, price-adjustment basis and SHA-256. The repository may contain deterministic synthetic fixtures, but no unlicensed market-data copy. The first version rejects duplicate, missing, non-positive or non-finite prices and unequal per-symbol date grids. A fixed universe is a **demonstration limitation**, not a survivorship-free market backtest.
- Every candidate sees price history only through signal date `t`. A signal calculated after the `t` close can first change the portfolio at the `t+1` close; its first measured return is from the `t+1` close to the `t+2` close. There is no same-close execution. Fees and slippage are charged on absolute weight changes.
- Dates split chronologically into train, development and sealed final evaluation. The final period is opened only after the candidate and selection rule are frozen. If a future-return label spans `H` sessions, adjacent selection windows require at least `H` sessions of separation. Earlier prices may warm up features; later labels and fitted statistics may not flow backwards.

## First deliverables

1. A zero-model deterministic price loader, baseline signals and backtester with future-data, timing, fee and input-validation tests.
2. A quant task adapter for the existing experiment tree, sandbox, model ledger and replay mechanisms. Research candidates run only on development data. Existing `bank-v1` tasks and their records stay unchanged.
3. A separate finalizer that evaluates the frozen candidate once, writes a sealed report, and refuses a repeat. Do not use `evaluation.orchestrate.execute_task` here: it calls `finish_task` after every task.
4. A copyable, key-free demo and documented Qwen-only comparison. Add model routing only after the deterministic and Qwen-only paths are auditable.

The report must show every candidate, including failures; the dataset, code and configuration hashes; net return, risk, turnover, wall time and model usage. Unknown model usage remains unknown, not zero.
