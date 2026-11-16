# Quant Research MVP: reproducible factor experiments

This is a research demonstration, not trading advice or evidence of live returns. The first milestone evaluates **one fixed universe at daily frequency**. A candidate is a signal function that ranks the assets; the system either rejects it or retains it with a complete experiment record. Positive returns are not a completion criterion. A key-free scripted run and a separate one-shot finalizer are implemented; no real Qwen quant result is claimed.

## Data and timing contract

- Input is a user-supplied CSV with `date,symbol,close` columns, plus a manifest recording its source, redistribution permission, price-adjustment basis and SHA-256. The repository may contain deterministic synthetic fixtures, but no unlicensed market-data copy. The first version rejects duplicate, missing, non-positive or non-finite prices and unequal per-symbol date grids. A fixed universe is a **demonstration limitation**, not a survivorship-free market backtest.
- Every candidate sees price history only through signal date `t`. A signal calculated after the `t` close can first change the portfolio at the `t+1` close; its first measured return is from the `t+1` close to the `t+2` close. There is no same-close execution. Fees and slippage are charged on absolute weight changes.
- Dates split chronologically into train, development and sealed final evaluation. The trusted parent validates the full CSV at load time, but neither the model nor the candidate worker receives sealed future prices during development; the final period is **scored** only after the candidate and selection rule are frozen. If a future-return label spans `H` sessions, adjacent selection windows require at least `H` sessions of separation. Earlier prices may warm up features; later labels and fitted statistics may not flow backwards.

The current label horizon is **two sessions** (`t` to `t+2`); the synthetic workflow leaves four sessions between its train/dev and dev/holdout result windows. A dataset manifest has this shape (replace the source, permission statement and digest with your actual data):
The current signal functions are rule-based and do not fit parameters on the train window; that window is reserved for historical context and future train-only fitting.

```json
{
  "schema": "quant_dataset/1",
  "csv": "prices.csv",
  "csv_sha256": "64 lowercase hexadecimal characters",
  "source": "origin and acquisition date",
  "license": "permission or restriction statement",
  "price_basis": "close adjustment and corporate-action policy",
  "universe_policy": "fixed_demo"
}
```

The CSV header is exactly `date,symbol,close`, sorted by date, with a complete price for every symbol on every date. `fixed_demo` is a deliberately narrow policy name: it does **not** establish point-in-time membership or survivorship-free results.

## Implemented workflow

1. A deterministic loader, reference signals and cost-aware backtester, plus future-data, timing and malformed-input tests.
2. `quant_research.run` prepares an isolated `quant_run/1`, invokes the existing experiment tree, ledger and library through `QuantHarness`, and freezes the highest accepted development node. `--mode scripted` is a zero-model engineering test; `--mode qwen` is a real-model dev entry point and requires an API key in the process environment. No model is called in the scripted example.
3. `quant_research.finalize_cli --evaluate-sealed-once` checks candidate, score, dataset, manifest and ledger hashes before a separate scorer opens the sealed interval. It creates an exclusive attempt marker before scoring and refuses any repeat, including after a failure. It is never called from `evaluation.orchestrate.execute_task`, which scores holdout after each old-style task.
4. The run directory retains every node, including rejected and failed proposals, and reports four disjoint token buckets. Unknown usage is flagged, with the total set to null. The test-only token price table cannot support a cash-cost claim.

The report shows every candidate, including failures; the dataset, code, configuration and ledger hashes; net return, risk, turnover, wall time and model usage. Unknown model usage remains unknown, not zero.

Real-market evaluation, a point-in-time universe, dividend/split verification, multiple independent periods and Qwen-vs-scripted research comparisons are future work. The synthetic holdout only validates the one-shot mechanism, not predictive quality.
