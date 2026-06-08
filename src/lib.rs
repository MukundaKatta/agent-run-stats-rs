/*!
agent-run-stats: lightweight run statistics collector for AI agent sessions.

Collect tool-call records, token usage, and cost during an agent run. Produce
a structured summary — including per-tool counts, average and p95 latency, and
the most-used tool — at the end. Zero overhead on the happy path: the collector
only allocates when you record.

# Example

```rust
use agent_run_stats::RunStats;

let mut stats = RunStats::new();
stats.record_tool_call("search", 120);
stats.record_tokens(500, 200, 0.005);

let summary = stats.summary();
assert_eq!(summary.total_tool_calls, 1);
assert_eq!(summary.tokens_in, 500);
assert_eq!(summary.total_tokens, 700);
assert!((summary.cost_usd - 0.005).abs() < 1e-9);

// Ship the summary to your observability backend as JSON.
let _json = summary.to_json();
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
    /// 95th-percentile latency (ms) per tool, computed with
    /// nearest-rank interpolation over the recorded durations.
    pub tool_p95_duration_ms: HashMap<String, f64>,
    pub most_used_tool: Option<String>,
    /// Total tokens (input + output) across all recorded LLM calls.
    pub total_tokens: u64,
}

impl RunSummary {
    /// Serialize the summary to a [`serde_json::Value`].
    ///
    /// All aggregate fields are included, so the JSON is a faithful
    /// representation of the summary that can be logged or shipped to an
    /// observability backend.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "duration_ms": self.duration_ms,
            "total_tool_calls": self.total_tool_calls,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "tool_call_counts": self.tool_call_counts,
            "tool_avg_duration_ms": self.tool_avg_duration_ms,
            "tool_p95_duration_ms": self.tool_p95_duration_ms,
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

    /// Record a tool call with its name and latency in milliseconds.
    pub fn record_tool_call(&mut self, name: &str, duration_ms: u64) {
        self.tool_calls.push(ToolCallStat {
            name: name.to_owned(),
            duration_ms,
        });
    }

    /// Record a tool call using a [`Duration`] instead of raw milliseconds.
    ///
    /// Convenient when timing with [`Instant::elapsed`]. Sub-millisecond
    /// durations are rounded down to `0`.
    ///
    /// ```
    /// use std::time::Duration;
    /// use agent_run_stats::RunStats;
    ///
    /// let mut stats = RunStats::new();
    /// stats.record_tool_call_duration("search", Duration::from_millis(120));
    /// assert_eq!(stats.summary().tool_avg_duration_ms["search"], 120.0);
    /// ```
    pub fn record_tool_call_duration(&mut self, name: &str, duration: Duration) {
        self.record_tool_call(name, duration.as_millis() as u64);
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

    /// Total tokens (input + output) across all recorded LLM calls.
    pub fn total_tokens(&self) -> u64 {
        self.tokens_in() + self.tokens_out()
    }

    /// Merge another collector's records into this one.
    ///
    /// Useful for aggregating stats collected on worker threads or in
    /// sub-runs. The timer of `self` is left untouched.
    pub fn merge(&mut self, other: &RunStats) {
        self.tool_calls.extend(other.tool_calls.iter().cloned());
        self.token_records
            .extend(other.token_records.iter().cloned());
    }

    /// Produce an aggregated summary.
    pub fn summary(&self) -> RunSummary {
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut durations: HashMap<String, Vec<u64>> = HashMap::new();
        for tc in &self.tool_calls {
            *counts.entry(tc.name.clone()).or_insert(0) += 1;
            durations
                .entry(tc.name.clone())
                .or_default()
                .push(tc.duration_ms);
        }
        let avg_duration: HashMap<String, f64> = durations
            .iter()
            .map(|(k, v)| {
                let avg = v.iter().sum::<u64>() as f64 / v.len() as f64;
                (k.clone(), avg)
            })
            .collect();

        let p95_duration: HashMap<String, f64> = durations
            .iter()
            .map(|(k, v)| (k.clone(), percentile(v, 95.0)))
            .collect();

        let most_used = counts
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.clone());

        let tokens_in = self.tokens_in();
        let tokens_out = self.tokens_out();

        RunSummary {
            duration_ms: self.elapsed_ms(),
            total_tool_calls: self.tool_calls.len(),
            tokens_in,
            tokens_out,
            cost_usd: self.cost_usd(),
            tool_call_counts: counts,
            tool_avg_duration_ms: avg_duration,
            tool_p95_duration_ms: p95_duration,
            most_used_tool: most_used,
            total_tokens: tokens_in + tokens_out,
        }
    }

    /// Reset all recorded stats (timer continues from original start).
    pub fn reset_stats(&mut self) {
        self.tool_calls.clear();
        self.token_records.clear();
    }
}

/// Nearest-rank percentile over a slice of samples.
///
/// `p` is a percentile in the range `[0, 100]`. Returns `0.0` for an empty
/// slice. The input is not required to be sorted.
fn percentile(samples: &[u64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<u64> = samples.to_vec();
    sorted.sort_unstable();
    let p = p.clamp(0.0, 100.0);
    // Nearest-rank: rank = ceil(p/100 * n), 1-based, clamped to [1, n].
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx] as f64
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

    #[test]
    fn record_tool_call_duration_uses_millis() {
        let mut s = RunStats::new();
        s.record_tool_call_duration("search", Duration::from_millis(120));
        assert_eq!(s.tool_call_count(), 1);
        assert_eq!(s.summary().tool_avg_duration_ms["search"], 120.0);
    }

    #[test]
    fn record_tool_call_duration_rounds_sub_millis_to_zero() {
        let mut s = RunStats::new();
        s.record_tool_call_duration("fast", Duration::from_micros(500));
        assert_eq!(s.summary().tool_avg_duration_ms["fast"], 0.0);
    }

    #[test]
    fn total_tokens_sums_in_and_out() {
        let mut s = RunStats::new();
        s.record_tokens(100, 50, 0.001);
        s.record_tokens(200, 100, 0.002);
        assert_eq!(s.total_tokens(), 450);
        assert_eq!(s.summary().total_tokens, 450);
    }

    #[test]
    fn merge_combines_records() {
        let mut a = RunStats::new();
        a.record_tool_call("search", 10);
        a.record_tokens(100, 50, 0.001);

        let mut b = RunStats::new();
        b.record_tool_call("search", 30);
        b.record_tool_call("fetch", 5);
        b.record_tokens(200, 100, 0.002);

        a.merge(&b);
        let sum = a.summary();
        assert_eq!(sum.total_tool_calls, 3);
        assert_eq!(sum.tool_call_counts["search"], 2);
        assert_eq!(sum.tokens_in, 300);
        assert_eq!(sum.tokens_out, 150);
        assert!((sum.cost_usd - 0.003).abs() < 1e-9);
    }

    #[test]
    fn summary_p95_single_sample() {
        let mut s = RunStats::new();
        s.record_tool_call("t", 42);
        assert_eq!(s.summary().tool_p95_duration_ms["t"], 42.0);
    }

    #[test]
    fn summary_p95_picks_high_tail() {
        let mut s = RunStats::new();
        for d in [10u64, 20, 30, 40, 50, 60, 70, 80, 90, 1000] {
            s.record_tool_call("t", d);
        }
        // Nearest-rank p95 over 10 samples -> rank ceil(9.5) = 10 -> the max.
        assert_eq!(s.summary().tool_p95_duration_ms["t"], 1000.0);
    }

    #[test]
    fn percentile_empty_is_zero() {
        assert_eq!(percentile(&[], 95.0), 0.0);
    }

    #[test]
    fn percentile_median() {
        // 1-based nearest rank for p50 over 5 elements -> rank ceil(2.5)=3.
        assert_eq!(percentile(&[1, 2, 3, 4, 5], 50.0), 3.0);
    }

    #[test]
    fn percentile_handles_unsorted_input() {
        assert_eq!(percentile(&[5, 1, 3, 2, 4], 100.0), 5.0);
        assert_eq!(percentile(&[5, 1, 3, 2, 4], 0.0), 1.0);
    }

    #[test]
    fn to_json_includes_avg_and_p95() {
        let mut s = RunStats::new();
        s.record_tool_call("t", 10);
        s.record_tool_call("t", 30);
        let j = s.summary().to_json();
        assert!(j["tool_avg_duration_ms"]["t"].as_f64().is_some());
        assert!(j["tool_p95_duration_ms"]["t"].as_f64().is_some());
        assert!(j["total_tokens"].as_u64().is_some());
    }
}
