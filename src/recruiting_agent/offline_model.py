"""Deterministic LangChain chat model used for offline runs and tests.

It implements the same contract as a real tool-calling LLM: given the conversation it
either emits ``tool_calls`` or a final answer composed only from tool results. That lets
the full LangGraph -> ToolNode -> MCP loop run reproducibly without an API key. It is a
stand-in for the LLM, not a claim about LLM quality; set RECRUITING_AGENT_LLM=anthropic
to run the same graph with Claude.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

SKILLS = [
    "python", "langgraph", "langchain", "mcp", "aws", "evaluation", "fastapi", "postgresql",
    "docker", "kubernetes", "pytorch", "mlflow", "typescript", "react", "graphql", "css", "sql",
    "spark", "airflow", "dbt", "llm", "rag", "guardrails", "terraform", "ci/cd", "scheduling",
    "ats", "communication",
]  # fmt: skip
TITLE_WORDS = ["llm", "backend", "frontend", "data", "devops", "platform", "conversational",
               "recruiting", "engineer"]  # fmt: skip
LOCATIONS = ["remote (us)", "remote (eu)", "austin", "new york", "toronto", "chicago", "remote"]

CAND_ID = re.compile(r"\bC-\d{3}\b", re.IGNORECASE)
JOB_ID = re.compile(r"\bJ-\d{3}\b", re.IGNORECASE)
CAND_REF = re.compile(
    r"\b(that|this|the same) (candidate|person|engineer)\b|\b(them|they|their|he|she|him|her)\b",
    re.IGNORECASE,
)
TOP_REF = re.compile(r"\b(top|first|best|strongest) (one|candidate|match|pick)\b", re.IGNORECASE)
SECOND_REF = re.compile(r"\b(second|runner[- ]up|next) (one|candidate)\b", re.IGNORECASE)
JOB_REF = re.compile(r"\b(that|this|the same|the) (job|role|position|opening)\b", re.IGNORECASE)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, dict):
            parts.append(block.get("text", ""))
        else:
            parts.append(str(block))
    return "".join(parts)


def _parse(content: Any) -> Any:
    raw = _text(content)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _call(name: str, **args: Any) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}


class _Memory:
    """What earlier turns established: focused candidate/job and the last shortlist."""

    def __init__(self, history: list[BaseMessage]):
        self.candidate: str | None = None
        self.job: str | None = None
        self.shortlist: list[str] = []
        self.last_tool: str | None = None
        for msg in history:
            if isinstance(msg, ToolMessage):
                self.last_tool = msg.name
                self._absorb(msg.name, _parse(msg.content))
            elif isinstance(msg, HumanMessage):
                text = _text(msg.content)
                if ids := CAND_ID.findall(text):
                    self.candidate = ids[-1].upper()
                if ids := JOB_ID.findall(text):
                    self.job = ids[-1].upper()

    def _absorb(self, tool: str | None, data: Any) -> None:
        if tool == "search_candidates" and isinstance(data, dict) and data.get("candidates"):
            self.shortlist = [c["id"] for c in data["candidates"]]
            self.candidate = self.shortlist[0]
        elif tool == "get_candidate" and isinstance(data, dict) and "id" in data:
            self.candidate = data["id"]
        elif tool == "get_job" and isinstance(data, dict) and "id" in data:
            self.job = data["id"]
        elif tool == "search_jobs" and isinstance(data, dict) and data.get("jobs"):
            self.job = data["jobs"][0]["id"]
        elif tool == "score_match" and isinstance(data, dict) and "job_id" in data:
            self.job = data["job_id"]


class OfflinePlannerModel(BaseChatModel):
    """Rule-based planner that speaks the LangChain tool-calling protocol."""

    @property
    def _llm_type(self) -> str:
        return "offline-planner"

    def bind_tools(self, tools: Any, **kwargs: Any) -> OfflinePlannerModel:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kw
    ) -> ChatResult:
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        prior, turn = messages[:last_human], messages[last_human + 1 :]
        question = _text(messages[last_human].content)
        memory = _Memory(prior)
        results = [m for m in turn if isinstance(m, ToolMessage)]
        if not results:
            message = self._plan(question, memory)
        else:
            message = self._follow_up(question, results) or self._answer(results)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- planning ----------------------------------------------------------------------

    def _plan(self, q: str, mem: _Memory) -> AIMessage:
        low = q.lower()
        cand_ids = [c.upper() for c in CAND_ID.findall(q)]
        job_ids = [j.upper() for j in JOB_ID.findall(q)]
        if not cand_ids:
            if SECOND_REF.search(q) and len(mem.shortlist) > 1:
                cand_ids = [mem.shortlist[1]]
            elif TOP_REF.search(q) and mem.shortlist:
                cand_ids = [mem.shortlist[0]]
            elif CAND_REF.search(q) and mem.candidate:
                cand_ids = [mem.candidate]
        if not job_ids and JOB_REF.search(q) and mem.job:
            job_ids = [mem.job]
        skills = [s for s in SKILLS if re.search(rf"(?<![\w/]){re.escape(s)}(?![\w/])", low)]

        if re.search(r"\b(policy|allowed|legal|lawful|eeo|compliant|can we ask|can i ask)\b", low):
            topic = "eeo"
            if re.search(r"privacy|contact|email|phone|personal data", low):
                topic = "data_privacy"
            elif re.search(r"\b(ai|automat\w*|score)\b", low):
                topic = "ai_use"
            elif re.search(r"screen\w*|interview|question", low):
                topic = "screening"
            return self._calls([_call("get_policy", topic=topic)])

        match_words = r"\b(match|fit|score|suitable|good for|qualified|compare|missing|gaps?)\b"
        ellipsis = re.search(r"\b(what|how) about\b", low) and mem.last_tool == "score_match"
        if re.search(match_words, low) or ellipsis:
            job = (job_ids or [mem.job])[0]
            cands = cand_ids or ([mem.candidate] if mem.candidate else [])
            if re.search(r"\b(both|shortlist|all of them)\b", low) and mem.shortlist:
                cands = mem.shortlist[:3]
            if not job or not cands:
                return AIMessage("Which candidate id (C-###) and job id (J-###) should I score?")
            return self._calls([_call("score_match", candidate_id=c, job_id=job) for c in cands])

        wants_people = re.search(
            r"\b(candidates?|people|talent|engineers? (who|with)|anyone|shortlist)\b", low
        )
        if wants_people:
            years = re.search(r"(\d+)\+?\s*(years?|yrs)", low)
            min_years = int(years.group(1)) if years else 0
            if job_ids and not skills:
                return self._calls([_call("get_job", job_id=job_ids[0])])
            if skills:
                return self._calls([_call("search_candidates", skills=skills, min_years=min_years)])

        if cand_ids and re.search(
            r"\b(about|details?|profile|background|notes|more|who is)\b|^C-", q
        ):
            return self._calls([_call("get_candidate", candidate_id=c) for c in cand_ids])
        if cand_ids:
            return self._calls([_call("get_candidate", candidate_id=c) for c in cand_ids])

        if job_ids:
            return self._calls([_call("get_job", job_id=j) for j in job_ids])

        if re.search(r"\b(jobs?|roles?|openings?|positions?|hiring for)\b", low):
            words = skills + [w for w in TITLE_WORDS if w != "engineer" and w in low]
            location = next((loc for loc in LOCATIONS if loc in low), "")
            return self._calls([_call("search_jobs", query=" ".join(words), location=location)])

        return AIMessage(
            "I can search jobs, find candidates by skills, show a candidate or job by id, score a "
            "candidate against a job, or explain hiring policy. What would you like to do?"
        )

    @staticmethod
    def _calls(calls: list[dict]) -> AIMessage:
        return AIMessage(content="", tool_calls=calls)

    def _follow_up(self, q: str, results: list[ToolMessage]) -> AIMessage | None:
        """Second hop: 'candidates for J-101' -> get_job, then search with its skills."""
        tools_used = {r.name for r in results}
        wants_people = re.search(r"\b(candidates?|people|talent|shortlist)\b", q, re.IGNORECASE)
        if wants_people and tools_used == {"get_job"}:
            job = _parse(results[-1].content)
            if isinstance(job, dict) and "skills" in job:
                return self._calls(
                    [_call("search_candidates", skills=job["skills"], min_years=job["min_years"])]
                )
        return None

    # -- answering ---------------------------------------------------------------------

    def _answer(self, results: list[ToolMessage]) -> AIMessage:
        lines: list[str] = []
        for r in results:
            lines.extend(_render(r.name, _parse(r.content)))
        tools = ", ".join(dict.fromkeys(r.name for r in results))
        lines.append(f"Sources: MCP tools {tools}.")
        return AIMessage("\n".join(lines))


def _render(tool: str | None, data: Any) -> list[str]:
    if isinstance(data, dict) and "error" in data:
        return [f"Lookup failed: {data['error']}."]
    if tool == "search_candidates":
        if not data["candidates"]:
            return ["No candidates matched those criteria."]
        out = ["Candidates ranked by skill overlap:"]
        for i, c in enumerate(data["candidates"], 1):
            out.append(
                f"{i}. {c['id']} {c['name']} - {c['headline']}, {c['years']} yrs, {c['location']};"
                f" skills: {', '.join(c['skills'])}"
            )
        return out
    if tool == "get_candidate":
        return [
            f"{data['id']} {data['name']} - {data['headline']}, {data['years']} yrs, "
            f"{data['location']}. Skills: {', '.join(data['skills'])}. Notes: {data['notes']}"
        ]
    if tool == "search_jobs":
        if not data["jobs"]:
            return ["No open jobs matched."]
        out = ["Matching jobs:"]
        for j in data["jobs"]:
            out.append(
                f"- {j['id']} {j['title']} ({j['location']}); skills: {', '.join(j['skills'])}"
            )
        return out
    if tool == "get_job":
        return [
            f"{data['id']} {data['title']} ({data['location']}), min {data['min_years']} yrs. "
            f"Skills: {', '.join(data['skills'])}. {data['description']}"
        ]
    if tool == "score_match":
        missing = ", ".join(data["missing_skills"]) or "none"
        return [
            f"{data['candidate_id']} vs {data['job_id']}: score {data['score']:.2f} "
            f"(matched: {', '.join(data['matched_skills']) or 'none'}; missing: {missing}; "
            f"{data['years']} of {data['min_years']} required yrs). {data['note']}"
        ]
    if tool == "get_policy":
        return [f"Policy ({data['topic']}): {data['text']}"]
    return [json.dumps(data)]
