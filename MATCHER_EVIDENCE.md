# Skill evidence map

This file shows where each skill tag is demonstrated in code, and how to check it. Everything is in this repository and runs in CI.

| Skill tag | Evidence | How to verify |
|---|---|---|
| Model Context Protocol (MCP) | [`mcp_server.py`](src/recruiting_agent/mcp_server.py): FastMCP server with 6 tools, a resource template, and a prompt; stdio and streamable-HTTP transports | `tests/test_mcp_server.py::test_real_stdio_mcp_round_trip` |
| LangChain | [`graph.py`](src/recruiting_agent/graph.py): `langchain-mcp-adapters` loads MCP tools as LangChain tools; [`offline_model.py`](src/recruiting_agent/offline_model.py) is a LangChain `BaseChatModel` that uses the tool-calling protocol | `tests/test_agent.py` |
| LangGraph | [`graph.py`](src/recruiting_agent/graph.py): `StateGraph`, conditional edges, `ToolNode`, `InMemorySaver` thread memory | `test_multi_turn_memory_resolves_references`, `test_threads_are_isolated` |
| LLM Agents / Agentic Frameworks | Multi-step plans (`get_job` → `search_candidates`), reference resolution across turns, bounded tool rounds | `test_two_hop_tool_plan`; eval category `multi_step_tools` |
| Guardrails / AI safety | [`guardrails.py`](src/recruiting_agent/guardrails.py): input, tool-output, and final-answer guards | `tests/test_guardrails.py`; the ablation in `results/eval_report.md` |
| Conversational evaluation | [`evals.py`](src/recruiting_agent/evals.py): multi-turn scenarios, trajectory/content/block/leak metrics, guarded vs unguarded | CI step "Conversational eval" |
| HR & Staffing domain | EEO, screening, and data-privacy policies enforced in the agent; recruiter-facing match scoring | `evals/conversations.json` categories `safety`, `policy`, `privacy` |
| Deployment | FastAPI, Dockerfile with health check, CI builds the image and calls `/chat` | `.github/workflows/ci.yml` job `docker` |

**What this does not show:** it has no real users, is not a production deployment, and uses only synthetic data. It shows engineering ability, not production tenure.
