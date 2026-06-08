//! Integration tests exercising the public API as an external consumer.

use std::time::Duration;

use agent_run_stats::RunStats;

#[test]
fn full_run_summary_is_consistent() {
    let mut stats = RunStats::new();

    stats.record_tool_call("search", 100);
    stats.record_tool_call_duration("search", Duration::from_millis(300));
    stats.record_tool_call("fetch", 50);

    stats.record_tokens(1_000, 400, 0.012);
    stats.record_tokens(500, 200, 0.006);

    let summary = stats.summary();

    assert_eq!(summary.total_tool_calls, 3);
    assert_eq!(summary.tool_call_counts["search"], 2);
    assert_eq!(summary.tool_call_counts["fetch"], 1);
    assert_eq!(summary.most_used_tool, Some("search".to_string()));

    assert_eq!(summary.tokens_in, 1_500);
    assert_eq!(summary.tokens_out, 600);
    assert_eq!(summary.total_tokens, 2_100);
    assert!((summary.cost_usd - 0.018).abs() < 1e-9);

    // search averaged (100 + 300) / 2 = 200ms.
    assert!((summary.tool_avg_duration_ms["search"] - 200.0).abs() < 1e-9);
    // p95 over two samples picks the slower one.
    assert_eq!(summary.tool_p95_duration_ms["search"], 300.0);
}

#[test]
fn json_round_trips_all_fields() {
    let mut stats = RunStats::new();
    stats.record_tool_call("a", 10);
    stats.record_tokens(100, 50, 0.001);

    let json = stats.summary().to_json();

    // Every aggregate field is present in the serialized form.
    for key in [
        "duration_ms",
        "total_tool_calls",
        "tokens_in",
        "tokens_out",
        "total_tokens",
        "cost_usd",
        "tool_call_counts",
        "tool_avg_duration_ms",
        "tool_p95_duration_ms",
        "most_used_tool",
    ] {
        assert!(json.get(key).is_some(), "missing key in JSON: {key}");
    }

    assert_eq!(json["total_tokens"].as_u64(), Some(150));
}

#[test]
fn merge_then_reset_behaves() {
    let mut main = RunStats::new();
    main.record_tool_call("x", 10);

    let mut worker = RunStats::new();
    worker.record_tool_call("y", 20);
    worker.record_tokens(10, 5, 0.0001);

    main.merge(&worker);
    assert_eq!(main.summary().total_tool_calls, 2);

    main.reset_stats();
    assert_eq!(main.tool_call_count(), 0);
    assert_eq!(main.total_tokens(), 0);
}
