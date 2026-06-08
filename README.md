# agent-run-stats

[![CI](https://github.com/MukundaKatta/agent-run-stats-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/MukundaKatta/agent-run-stats-rs/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/agent-run-stats.svg)](https://crates.io/crates/agent-run-stats)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

A lightweight run-statistics collector for AI agent sessions, written in Rust.

`agent-run-stats` records tool-call latencies, token usage, and cost while an
agent run is in flight, then produces a structured summary at the end. It is
dependency-light (only `serde_json`), allocation-free on the happy path until
you actually record something, and easy to wire into any agent loop or
observability pipeline.

## Features

- **Tool-call tracking** — record each tool invocation with its name and latency.
- **Token & cost accounting** — accumulate input/output tokens and USD cost across LLM calls.
- **Rich summary** — total and per-tool call counts, average **and p95** latency
  per tool, total tokens, total cost, and the most-used tool.
- **JSON output** — `RunSummary::to_json()` emits a complete `serde_json::Value`
  ready to log or ship to a backend.
- **Mergeable** — combine collectors from worker threads or sub-runs with `merge`.

## Installation

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
agent-run-stats = "0.1"
```

Or with cargo:

```sh
cargo add agent-run-stats
```

## Usage

```rust
use std::time::Duration;
use agent_run_stats::RunStats;

// The run timer starts when the collector is created.
let mut stats = RunStats::new();

// Record tool calls (raw milliseconds, or a Duration).
stats.record_tool_call("search", 120);
stats.record_tool_call_duration("search", Duration::from_millis(300));
stats.record_tool_call("fetch", 50);

// Record token usage and cost for each LLM call.
stats.record_tokens(1_000, 400, 0.012);
stats.record_tokens(500, 200, 0.006);

// Produce an aggregated summary at the end of the run.
let summary = stats.summary();

assert_eq!(summary.total_tool_calls, 3);
assert_eq!(summary.tool_call_counts["search"], 2);
assert_eq!(summary.most_used_tool, Some("search".to_string()));
assert_eq!(summary.total_tokens, 2_100);

// Average and p95 latency per tool.
assert_eq!(summary.tool_avg_duration_ms["search"], 200.0);
assert_eq!(summary.tool_p95_duration_ms["search"], 300.0);

// Serialize to JSON for logging / observability.
println!("{}", summary.to_json());
```

### Merging collectors

When work happens across threads or sub-runs, collect independently and merge:

```rust
use agent_run_stats::RunStats;

let mut main = RunStats::new();
main.record_tool_call("plan", 10);

let mut worker = RunStats::new();
worker.record_tool_call("search", 25);
worker.record_tokens(100, 50, 0.001);

main.merge(&worker);
assert_eq!(main.summary().total_tool_calls, 2);
```

## API

### `RunStats`

The collector for a single agent run.

| Method | Description |
| --- | --- |
| `RunStats::new()` | Create a collector; the run timer starts now. |
| `record_tool_call(name, duration_ms)` | Record a tool call with latency in ms. |
| `record_tool_call_duration(name, Duration)` | Record a tool call using a `Duration`. |
| `record_tokens(tokens_in, tokens_out, cost_usd)` | Record token usage and cost for one LLM call. |
| `elapsed_ms()` | Milliseconds elapsed since the run started. |
| `tool_call_count()` | Number of tool calls recorded so far. |
| `tokens_in()` / `tokens_out()` / `total_tokens()` | Aggregate token counts. |
| `cost_usd()` | Total accumulated cost in USD. |
| `merge(&other)` | Merge another collector's records into this one. |
| `summary()` | Produce an aggregated [`RunSummary`]. |
| `reset_stats()` | Clear recorded stats (the timer keeps running). |

### `RunSummary`

The aggregated result of a run. Public fields include `duration_ms`,
`total_tool_calls`, `tokens_in`, `tokens_out`, `total_tokens`, `cost_usd`,
`tool_call_counts`, `tool_avg_duration_ms`, `tool_p95_duration_ms`, and
`most_used_tool`. Call `to_json()` to obtain a complete `serde_json::Value`.

## Development

```sh
cargo build
cargo test
cargo fmt --check
cargo clippy --all-targets -- -D warnings
```

## License

Licensed under the [MIT License](LICENSE).
