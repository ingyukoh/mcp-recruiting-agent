"""Build the GitHub Pages evidence site (docs/index.html) from results/eval_report.json.

Every number and transcript on the page comes from the generated report.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/ingyukoh/mcp-recruiting-agent"
BLOB = f"{REPO}/blob/main"

EVIDENCE = [
    ("Model Context Protocol (MCP)", "MCP server with 6 tools, a resource template, and a prompt; "
     "stdio and streamable-HTTP transports", "src/recruiting_agent/mcp_server.py"),
    ("LangChain", "MCP tools loaded as LangChain tools via langchain-mcp-adapters; the offline "
     "planner is a LangChain BaseChatModel that emits tool calls", "src/recruiting_agent/graph.py"),
    ("LangGraph", "StateGraph with guard nodes, ToolNode loop, conditional edges, per-thread "
     "checkpointed memory", "src/recruiting_agent/graph.py"),
    ("LLM Agents / Agentic Frameworks", "Multi-step tool plans (get_job then search_candidates), "
     "reference resolution across turns, bounded tool rounds", "src/recruiting_agent/offline_model.py"),
    ("Guardrails", "Input, tool-output, and final-answer guards: injection, EEO screening, PII, "
     "grounding of cited ids", "src/recruiting_agent/guardrails.py"),
    ("Conversational evaluation", "Multi-turn scenarios with trajectory, content, block, and leak "
     "checks, plus a guardrail ablation", "src/recruiting_agent/evals.py"),
    ("Packaging", "FastAPI service and Docker image with a health check", "Dockerfile"),
]  # fmt: skip

SHOWCASE = [
    ("Multi-turn memory", "shortlist-then-score", None),
    ("Two-hop tool plan", "job-to-shortlist", None),
    ("Indirect prompt injection in retrieved data", "indirect-injection", 1),
    ("PII in tool output", "pii-in-data", 1),
    ("Discriminatory screening request", "discrimination-requests", None),
]

METRIC_ROWS = [
    ("Conversation success", "conversation_success", "pct"),
    ("Turn pass rate", "turn_pass_rate", "pct"),
    ("Tool-trajectory accuracy", "tool_trajectory_accuracy", "pct"),
    ("Block precision", "block_precision", "pct"),
    ("Block recall", "block_recall", "pct"),
    ("Turns leaking PII", "pii_leak_turns", "int"),
    ("Turns echoing injected instructions", "injection_leak_turns", "int"),
    ("Turns citing ungrounded ids", "ungrounded_id_turns", "int"),
    ("Latency p50 (ms, offline model)", "latency_ms_p50", "ms"),
]

e = html.escape


def fmt(value, kind: str) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:.0f}"
    return str(value)


def metrics_table(report: dict) -> str:
    g, u = report["variants"]["guarded"]["metrics"], report["variants"]["unguarded"]["metrics"]
    rows = "".join(
        f"<tr><td>{e(label)}</td><td class=num>{fmt(g[k], kind)}</td>"
        f"<td class='num muted'>{fmt(u[k], kind)}</td></tr>"
        for label, k, kind in METRIC_ROWS
    )
    return (
        "<table><thead><tr><th>Metric</th><th class=num>Guardrails on</th>"
        f"<th class=num>Guardrails off</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def category_table(report: dict) -> str:
    g = report["variants"]["guarded"]["metrics"]["success_by_category"]
    u = report["variants"]["unguarded"]["metrics"]["success_by_category"]
    rows = "".join(
        f"<tr><td>{e(c.replace('_', ' '))}</td><td class=num>{g[c]}</td>"
        f"<td class='num muted'>{u[c]}</td></tr>"
        for c in g
    )
    return (
        "<table><thead><tr><th>Scenario category</th><th class=num>On</th>"
        f"<th class=num>Off</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def transcript(report: dict, title: str, conv_id: str, contrast_turn: int | None) -> str:
    guarded = [t for t in report["variants"]["guarded"]["turns"] if t["conversation"] == conv_id]
    unguarded = {
        t["turn"]: t
        for t in report["variants"]["unguarded"]["turns"]
        if t["conversation"] == conv_id
    }
    parts = [f"<article class=convo><h3>{e(title)}</h3>"]
    for t in guarded:
        tools = " → ".join(t["tools"]) or "no tools"
        events = "".join(f"<span class=tag>{e(ev)}</span>" for ev in t["guard_events"])
        status = "blocked" if t["blocked"] else tools
        parts.append(
            f"<div class=turn><p class=user>{e(t['user'])}</p>"
            f"<p class=meta><code>{e(status)}</code>{events}</p>"
            f"<pre class=answer>{e(t['answer'])}</pre>"
        )
        if contrast_turn == t["turn"] and t["turn"] in unguarded:
            parts.append(
                "<details><summary>Same turn with guardrails off</summary>"
                f"<pre class='answer bad'>{e(unguarded[t['turn']]['answer'])}</pre></details>"
            )
        parts.append("</div>")
    parts.append("</article>")
    return "".join(parts)


ARCH_SVG = """<svg viewBox="0 0 760 210" role="img" aria-label="Agent graph: user message to input
guard, agent and MCP tool loop with tool-output guard, then output guard" class=arch>
<defs><marker id=a viewBox="0 0 10 10" refX=9 refY=5 markerWidth=7 markerHeight=7 orient=auto>
<path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>
<g font-size=13 text-anchor=middle>
<rect x=10 y=80 width=100 height=44 rx=8 class=node /><text x=60 y=107>User turn</text>
<rect x=150 y=80 width=110 height=44 rx=8 class="node guard"/><text x=205 y=107>input_guard</text>
<rect x=300 y=80 width=110 height=44 rx=8 class=node /><text x=355 y=101>agent</text>
<text x=355 y=116 class=small>LangChain chat model</text>
<rect x=300 y=10 width=110 height=44 rx=8 class=node /><text x=355 y=31>ToolNode</text>
<text x=355 y=46 class=small>LangChain MCP tools</text>
<rect x=470 y=10 width=150 height=44 rx=8 class="node mcp"/><text x=545 y=31>MCP server</text>
<text x=545 y=46 class=small>stdio · 6 tools</text>
<rect x=300 y=156 width=110 height=44 rx=8 class="node guard"/><text x=355 y=176>tool_output</text>
<text x=355 y=191 class=small>_guard</text>
<rect x=470 y=80 width=120 height=44 rx=8 class="node guard"/><text x=530 y=107>output_guard</text>
<rect x=640 y=80 width=110 height=44 rx=8 class=node /><text x=695 y=107>Answer</text>
</g>
<g stroke="currentColor" fill=none stroke-width=1.4 marker-end="url(#a)">
<path d="M110 102H148"/><path d="M260 102H298"/><path d="M355 80V56"/>
<path d="M410 32H468"/><path d="M468 40H412"/>
<path d="M300 32C250 32 250 178 298 178"/><path d="M410 178C450 178 450 110 412 110"/>
<path d="M410 96H468"/><path d="M590 102H638"/>
<path d="M205 124C205 200 700 200 695 126" stroke-dasharray="4 4"/>
</g><text x=450 y=207 font-size=11 text-anchor=middle class=small>blocked turns skip the agent
</text></svg>"""


def build(report: dict) -> str:
    g = report["variants"]["guarded"]["metrics"]
    u = report["variants"]["unguarded"]["metrics"]
    evidence_rows = "".join(
        f"<tr><td><strong>{e(skill)}</strong></td><td>{e(what)}</td>"
        f"<td><a href='{BLOB}/{path}'><code>{e(path)}</code></a></td></tr>"
        for skill, what, path in EVIDENCE
    )
    showcase = "".join(transcript(report, *s) for s in SHOWCASE)
    return f"""<!doctype html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MCP Recruiting Agent</title>
<meta name=description content="Guarded multi-turn recruiting agent built on an MCP server,
LangChain MCP adapters, and LangGraph, with conversational evaluation.">
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6862;--line:#e3e0d8;--card:#ffffff;--accent:#1f5f8b;
--good:#2d6a3e;--bad:#9b2c2c;--guard:#f3ead6;--mcp:#e1edf5;--code:#f2f0ea}}
@media (prefers-color-scheme:dark){{:root:not([data-theme=light]){{--bg:#141413;--fg:#ecebe6;
--muted:#9c9990;--line:#2e2d2a;--card:#1b1b19;--accent:#7fb3d9;--good:#7cc08f;--bad:#e58b8b;
--guard:#3a3222;--mcp:#1d3140;--code:#23221f}}}}
:root[data-theme=dark]{{--bg:#141413;--fg:#ecebe6;--muted:#9c9990;--line:#2e2d2a;--card:#1b1b19;
--accent:#7fb3d9;--good:#7cc08f;--bad:#e58b8b;--guard:#3a3222;--mcp:#1d3140;--code:#23221f}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 ui-sans-serif,system-ui,
-apple-system,"Segoe UI",sans-serif}}
main{{max-width:960px;margin:0 auto;padding:48px 16px 80px}}
h1{{font-size:clamp(28px,5vw,40px);line-height:1.15;margin:0 0 12px;letter-spacing:-.01em}}
h2{{font-size:22px;margin:56px 0 12px;padding-top:8px;border-top:1px solid var(--line)}}
h3{{font-size:17px;margin:0 0 12px}}
p{{margin:0 0 12px}} a{{color:var(--accent)}}
.lede{{font-size:18px;color:var(--muted);max-width:720px}}
.links{{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0 0}}
.links a{{border:1px solid var(--line);background:var(--card);padding:6px 12px;border-radius:6px;
text-decoration:none}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:32px 0 0}}
.stat{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px}}
.stat b{{display:block;font-size:26px;font-variant-numeric:tabular-nums}}
.stat span{{color:var(--muted);font-size:14px}}
.scope{{border-left:3px solid var(--accent);background:var(--card);padding:12px 16px;margin:24px 0 0;
font-size:15px}}
.table-wrap{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse;font-size:15px;background:var(--card)}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-weight:600}} .num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.muted{{color:var(--muted)}}
code{{font:13px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:1px 5px;
border-radius:4px}}
pre{{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code);padding:12px;
border-radius:6px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;margin:8px 0}}
.arch{{width:100%;height:auto;color:var(--fg)}}
.arch .node{{fill:var(--card);stroke:var(--line)}} .arch .guard{{fill:var(--guard)}}
.arch .mcp{{fill:var(--mcp)}} .arch text{{fill:var(--fg)}} .arch .small{{fill:var(--muted);font-size:11px}}
.convo{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:18px;margin:16px 0}}
.turn{{padding:10px 0;border-top:1px solid var(--line)}} .turn:first-of-type{{border-top:0}}
.user{{font-weight:600;margin:0}} .user::before{{content:"User: ";color:var(--muted);font-weight:400}}
.meta{{margin:4px 0 0;font-size:13px}}
.tag{{display:inline-block;margin-left:6px;font-size:12px;padding:0 6px;border-radius:4px;
background:var(--guard)}}
.answer.bad{{border-left:3px solid var(--bad)}}
details summary{{cursor:pointer;color:var(--bad);font-size:14px}}
.two{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
@media (max-width:720px){{.two{{grid-template-columns:1fr}}}}
footer{{margin-top:56px;color:var(--muted);font-size:14px}}
</style></head><body><main>
<p class=muted>Portfolio project · Ingyu Koh</p>
<h1>MCP Recruiting Agent</h1>
<p class=lede>A guarded, multi-turn recruiting assistant. An <strong>MCP server</strong> exposes
hiring data as tools, <strong>LangChain MCP adapters</strong> turn them into LangChain tools, and a
<strong>LangGraph</strong> agent uses them across conversations. The whole loop is measured with
conversational evals, including a run with the guardrails removed.</p>
<div class=links><a href="{REPO}">Source on GitHub</a>
<a href="{REPO}/blob/main/results/eval_report.md">Generated eval report</a>
<a href="{REPO}/blob/main/MATCHER_EVIDENCE.md">Skill → code map</a></div>

<div class=stats>
<div class=stat><b>{report["conversations"]} / {report["turns"]}</b><span>conversations / turns</span></div>
<div class=stat><b>{g["conversation_success"] * 100:.0f}%</b><span>conversation success, guarded</span></div>
<div class=stat><b>{u["conversation_success"] * 100:.0f}%</b><span>same suite, guardrails off</span></div>
<div class=stat><b>{g["pii_leak_turns"] + g["injection_leak_turns"]} vs
{u["pii_leak_turns"] + u["injection_leak_turns"]}</b><span>PII + injection leaks, on vs off</span></div>
</div>

<p class=scope><strong>Scope, stated plainly:</strong> this is a portfolio system on synthetic data,
not a client deployment. The default model is a deterministic LangChain chat model so offline
evaluations are reproducible without API keys; the scenarios were written alongside it, so the guarded score is a
regression baseline, not a claim of general LLM quality. The informative number is the ablation.
Set <code>RECRUITING_AGENT_LLM=anthropic</code> to run the same graph and evals with Claude.</p>

<h2>Architecture</h2>
{ARCH_SVG}
<p class=muted>Guards run at three boundaries: user input, data returned by tools (indirect
injection), and the final answer (PII redaction and a check that every cited C-/J- id came from a
tool result in this conversation).</p>

<h2>Skill evidence</h2>
<div class=table-wrap><table><thead><tr><th>Skill</th><th>What the code does</th><th>Where</th>
</tr></thead><tbody>{evidence_rows}</tbody></table></div>

<h2>Evaluation results</h2>
<p>Generated {e(report["generated_at"])} by <code>python -m recruiting_agent.evals</code> against the
real MCP server over stdio. Run the same command locally to regenerate the report.</p>
<div class=two><div class=table-wrap>{metrics_table(report)}</div>
<div class=table-wrap>{category_table(report)}</div></div>

<h2>Transcripts from the eval run</h2>
<p class=muted>Verbatim output of the guarded run. Where it matters, expand to see the same turn
with guardrails off.</p>
{showcase}

<h2>Reproduce</h2>
<pre>git clone {REPO}
cd mcp-recruiting-agent
python -m venv .venv &amp;&amp; source .venv/bin/activate
pip install -e '.[dev]'
pytest -q                            # unit + end-to-end over real MCP stdio
python -m recruiting_agent.evals     # writes results/eval_report.md
uvicorn recruiting_agent.api:app     # POST /chat {{"thread_id": "...", "message": "..."}}</pre>
<p>Use the MCP server from any MCP client (for example Claude Desktop):</p>
<pre>{{"mcpServers": {{"recruiting": {{"command": "python",
  "args": ["-m", "recruiting_agent.mcp_server"]}}}}}}</pre>

<footer>Built by Ingyu Koh · Page generated from <code>results/eval_report.json</code> by
<code>scripts/build_site.py</code> · MIT License</footer>
</main></body></html>
"""


def main() -> None:
    report = json.loads((ROOT / "results" / "eval_report.json").read_text())
    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(build(report))
    (ROOT / "docs" / ".nojekyll").write_text("")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
