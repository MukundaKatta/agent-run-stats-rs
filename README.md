# agent-run-stats

A lightweight run-statistics collector for AI agent sessions, written in Rust.

`agent-run-stats` lets you record tool calls, token usage, and cost during a
single agent run, then produce a structured summary when the run finishes. It
is designed for low overhead on the happy path — it only allocates when you
record something — and has a single small dependency (`serde_json`) for JSON
output.

## What it does

During an agent run you typically want to know things like: how many tool
calls happened, which tool was used most, how long each tool took on average,
how many tokens went in and out, and what the run cost. This crate collects
those records and aggregates them into a `RunSummary` that can be inspected
directly or serialized to JSON.

Key pieces of the API:

- **`RunStats`** — the collector. Starts a timer on construction.
  - `record_tool_call(name, duration_ms)` — log a tool invocation and its latency.
  - `record_tokens(tokens_in, tokens_out, cost_usd)` — log token usage and cost for one LLM call.
  - `elapsed_ms()`, `tool_call_count()`, `tokens_in()`, `tokens_out()`, `cost_usd()` — live accessors.
  - `summary()` — produce an aggregated `RunSummary`.
  - `reset_stats()` — clear recorded records (the run timer keeps running).
- **`RunSummary`** — aggregated results, including per-tool call counts,
  per-tool average durations, and the most-used tool. Call `to_json()` to get a
  `serde_json::Value`.
- **`ToolCallStat`** / **`TokenRecord`** — the individual record types.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
agent-run-stats = "0.1"
```

Or, while it is developed against this repository:

```toml
[dependencies]
agent-run-stats = { git = "https://github.com/MukundaKatta/agent-run-stats-rs" }
```

## Usage

```rust
use agent_run_stats::RunStats;

let mut stats = RunStats::new();

// Record tool calls with their latency in milliseconds.
stats.record_tool_call("search", 120);
stats.record_tool_call("search", 80);
stats.record_tool_call("fetch", 200);

// Record token usage and cost per LLM call.
stats.record_tokens(500, 200, 0.005);

// Produce an aggregated summary at the end of the run.
let summary = stats.summary();

assert_eq!(summary.total_tool_calls, 3);
assert_eq!(summary.tool_call_counts["search"], 2);
assert_eq!(summary.most_used_tool, Some("search".to_string()));
assert_eq!(summary.tokens_in, 500);

// Serialize to JSON for logging or transport.
let json = summary.to_json();
println!("{json}");
```

## Tech stack

- **Language:** Rust (edition 2021)
- **Dependencies:** [`serde_json`](https://crates.io/crates/serde_json) for JSON output
- **Standard library:** `std::time` for run timing, `std::collections::HashMap` for per-tool aggregation

## Development

Build, test, and lint with the standard Cargo toolchain:

```sh
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The test suite lives inline in `src/lib.rs` and covers recording, aggregation,
summary output, and JSON serialization.

## License

Licensed under the MIT License. See the `license` field in `Cargo.toml`.
