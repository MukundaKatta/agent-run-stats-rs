/*!
agent-run-stats: lightweight run statistics collector for AI agent sessions.

Collect tool-call records, token usage, and cost during an agent run. Produce
a structured summary at the end. Zero overhead on the happy path — only
allocates when you record.

```rust
use agent_run_stats::RunStats;

let mut stats = RunStats::new();
stats.record_tool_call("search", 120);
stats.record_tokens(500, 200, 0.005);

let summary = stats.summary();
assert_eq!(summary.total_tool_calls, 1);
assert_eq!(summary.tokens_in, 500);
assert!((summary.cost_usd - 0.005).abs() < 1e-9);
```
*/

use serde_json::Value;
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---- ToolCallStat ---------------------------------------------------------

/// One tool-call record.
#[derive(Debug, Clone)]
pub struct ToolCallStat {
    pub name: String,
    pub duration_ms: u64,
}

// ---- TokenRecord ----------------------------------------------------------

/// One token usage record.
#[derive(Debug, Clone)]
pub struct TokenRecord {
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd: f64,
}

// ---- RunSummary -----------------------------------------------------------

/// Aggregated stats for the run.
#[derive(Debug, Clone)]
pub struct RunSummary {
    pub duration_ms: u64,
    pub total_tool_calls: usize,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub cost_usd: f64,
    pub tool_call_counts: HashMap<String, usize>,
    pub tool_avg_duration_ms: HashMap<String, f64>,
    pub most_used_tool: Option<String>,
}

impl RunSummary {
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "duration_ms": self.duration_ms,
            "total_tool_calls": self.total_tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "tool_call_counts": self.tool_call_counts,
            "most_used_tool": self.most_used_tool,
        })
    }
}

// ---- RunStats -------------------------------------------------------------

/// Collects stats during a single agent run.
pub struct RunStats {
    start: Instant,
    tool_calls: Vec<ToolCallStat>,
    token_records: Vec<TokenRecord>,
}

impl Default for RunStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunStats {
    /// Create a new stats collector. The run timer starts here.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            tool_calls: Vec::new(),
            token_records: Vec::new(),
        }
    }

    /// Record a tool call with its name and latency.
    pub fn record_tool_call(&mut self, name: &str, duration_ms: u64) {
        self.tool_calls.push(ToolCallStat {
            name: name.to_owned(),
            duration_ms,
        });
    }

    /// Record token usage and cost for one LLM call.
    pub fn record_tokens(&mut self, tokens_in: u64, tokens_out: u64, cost_usd: f64) {
        self.token_records.push(TokenRecord {
            tokens_in,
            tokens_out,
            cost_usd,
        });
    }

    /// Elapsed milliseconds since the run started.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Number of tool calls recorded so far.
    pub fn tool_call_count(&self) -> usize {
        self.tool_calls.len()
    }

    /// Total tokens in (sum of all recorded calls).
    pub fn tokens_in(&self) -> u64 {
        self.token_records.iter().map(|r| r.tokens_in).sum()
    }

    /// Total tokens out.
    pub fn tokens_out(&self) -> u64 {
        self.token_records.iter().map(|r| r.tokens_out).sum()
    }

    /// Total cost in USD.
    pub fn cost_usd(&self) -> f64 {
        self.token_records.iter().map(|r| r.cost_usd).sum()
    }

    /// Produce an aggregated summary.
    pub fn summary(&self) -> RunSummary {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut durations: HashMap<String, Vec<u64>> = HashMap::new();
        for tc in &self.tool_calls {
            *counts.entry(tc.name.clone()).or_insert(0) += 1;
            durations.entry(tc.name.clone()).or_default().push(tc.duration_ms);
        }
        let avg_duration: HashMap<String, f64> = durations
            .iter()
            .map(|(k, v)| {
                let avg = v.iter().sum::<u64>() as f64 / v.len() as f64;
                (k.clone(), avg)
            })
            .collect();

        let most_used = counts
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.clone());

        RunSummary {
            duration_ms: self.elapsed_ms(),
            total_tool_calls: self.tool_calls.len(),
            tokens_in: self.tokens_in(),
            tokens_out: self.tokens_out(),
            cost_usd: self.cost_usd(),
            tool_call_counts: counts,
            tool_avg_duration_ms: avg_duration,
            most_used_tool: most_used,
        }
    }

    /// Reset all recorded stats (timer continues from original start).
    pub fn reset_stats(&mut self) {
        self.tool_calls.clear();
        self.token_records.clear();
    }
}

impl std::fmt::Debug for RunStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStats")
            .field("elapsed_ms", &self.elapsed_ms())
            .field("tool_calls", &self.tool_calls.len())
            .field("token_records", &self.token_records.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_has_zero_counts() {
        let s = RunStats::new();
        assert_eq!(s.tool_call_count(), 0);
        assert_eq!(s.tokens_in(), 0);
        assert_eq!(s.tokens_out(), 0);
        assert_eq!(s.cost_usd(), 0.0);
    }

    #[test]
    fn record_tool_call() {
        let mut s = RunStats::new();
        s.record_tool_call("search", 50);
        assert_eq!(s.tool_call_count(), 1);
    }

    #[test]
    fn record_tokens() {
        let mut s = RunStats::new();
        s.record_tokens(100, 50, 0.001);
        assert_eq!(s.tokens_in(), 100);
        assert_eq!(s.tokens_out(), 50);
        assert!((s.cost_usd() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn multiple_token_records_summed() {
        let mut s = RunStats::new();
        s.record_tokens(100, 50, 0.001);
        s.record_tokens(200, 100, 0.002);
        assert_eq!(s.tokens_in(), 300);
        assert_eq!(s.tokens_out(), 150);
        assert!((s.cost_usd() - 0.003).abs() < 1e-9);
    }

    #[test]
    fn elapsed_ms_positive() {
        let s = RunStats::new();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(s.elapsed_ms() > 0);
    }

    #[test]
    fn summary_total_tool_calls() {
        let mut s = RunStats::new();
        s.record_tool_call("a", 10);
        s.record_tool_call("b", 20);
        assert_eq!(s.summary().total_tool_calls, 2);
    }

    #[test]
    fn summary_counts_by_name() {
        let mut s = RunStats::new();
        s.record_tool_call("search", 10);
        s.record_tool_call("search", 20);
        s.record_tool_call("fetch", 30);
        let sum = s.summary();
        assert_eq!(sum.tool_call_counts["search"], 2);
        assert_eq!(sum.tool_call_counts["fetch"], 1);
    }

    #[test]
    fn summary_most_used_tool() {
        let mut s = RunStats::new();
        s.record_tool_call("a", 10);
        s.record_tool_call("a", 20);
        s.record_tool_call("b", 30);
        assert_eq!(s.summary().most_used_tool, Some("a".to_string()));
    }

    #[test]
    fn summary_most_used_none_when_empty() {
        let s = RunStats::new();
        assert!(s.summary().most_used_tool.is_none());
    }

    #[test]
    fn summary_avg_duration() {
        let mut s = RunStats::new();
        s.record_tool_call("t", 10);
        s.record_tool_call("t", 30);
        let sum = s.summary();
        assert!((sum.tool_avg_duration_ms["t"] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn summary_tokens_match() {
        let mut s = RunStats::new();
        s.record_tokens(500, 200, 0.005);
        let sum = s.summary();
        assert_eq!(sum.tokens_in, 500);
        assert_eq!(sum.tokens_out, 200);
        assert!((sum.cost_usd - 0.005).abs() < 1e-9);
    }

    #[test]
    fn reset_stats_clears_records() {
        let mut s = RunStats::new();
        s.record_tool_call("t", 10);
        s.record_tokens(100, 50, 0.01);
        s.reset_stats();
        assert_eq!(s.tool_call_count(), 0);
        assert_eq!(s.tokens_in(), 0);
    }

    #[test]
    fn summary_to_json_has_fields() {
        let mut s = RunStats::new();
        s.record_tool_call("t", 10);
        s.record_tokens(100, 50, 0.001);
        let j = s.summary().to_json();
        assert!(j["total_tool_calls"].as_u64().unwrap() > 0);
        assert!(j["tokens_in"].as_u64().unwrap() > 0);
    }

    #[test]
    fn debug_format() {
        let s = RunStats::new();
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("RunStats"));
    }
}
