# Telemetry & Performance Tracking

Telemetry metrics logged per query include:
- Retrieval latency (ms)
- Generation latency (ms)
- Total query latency (ms)
- Chunk count retrieved
- Token usage (prompt tokens, completion tokens, total tokens)

The telemetry logger calculates p50 and p95 percentile performance statistics over query history.
