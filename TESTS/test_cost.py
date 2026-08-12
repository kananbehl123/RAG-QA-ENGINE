from src.eval.cost_model import CostAnalyzer


def test_cost_calculation():
    report = CostAnalyzer.generate_cost_report()
    assert "100K" in report
    assert "1M" in report
    assert "10M" in report

    lance_1m = report["1M"]["lancedb"]
    pinecone_1m = report["1M"]["pinecone"]

    assert lance_1m["monthly_total_cost_usd"] < pinecone_1m["monthly_total_cost_usd"]
    assert report["1M"]["savings_percentage"] > 50.0
