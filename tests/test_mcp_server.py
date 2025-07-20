from langchain_mcp_adapters.client import MultiServerMCPClient

from recruiting_agent import mcp_server as srv
from recruiting_agent.graph import mcp_connection


def test_contact_fields_never_leave_server():
    for cand in srv.candidates():
        assert not {"email", "phone"} & set(srv.get_candidate(cand["id"]))


def test_search_candidates_ranks_by_overlap_then_years():
    ids = [c["id"] for c in srv.search_candidates(["langgraph", "mcp"])["candidates"]]
    assert ids[:2] == ["C-201", "C-211"]
    assert "C-211" not in [c["id"] for c in srv.search_candidates(["mcp"], 3)["candidates"]]


def test_score_match_is_transparent():
    result = srv.score_match("C-203", "J-101")
    assert result["missing_skills"] == ["langgraph", "mcp"]
    assert result["score"] == round(0.7 * 4 / 6 + 0.3 * 3 / 5, 2)


def test_unknown_ids_return_errors():
    assert "error" in srv.get_job("J-000")
    assert "error" in srv.score_match("C-999", "J-101")
    assert "topics" in srv.get_policy("nope")


async def test_real_stdio_mcp_round_trip():
    client = MultiServerMCPClient({"recruiting": mcp_connection()})
    tools = {t.name: t for t in await client.get_tools()}
    assert set(tools) == {
        "search_jobs",
        "get_job",
        "search_candidates",
        "get_candidate",
        "score_match",
        "get_policy",
    }
    out = await tools["get_policy"].ainvoke({"topic": "eeo"})
    assert "protected" in str(out) or "age" in str(out)
