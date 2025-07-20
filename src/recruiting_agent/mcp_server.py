"""MCP server exposing a synthetic recruiting dataset as tools and resources.

Run over stdio (default, used by the agent):   python -m recruiting_agent.mcp_server
Run over streamable HTTP (for remote clients): python -m recruiting_agent.mcp_server --http
"""

from __future__ import annotations

import argparse
import json
from functools import cache
from importlib import resources
from typing import Any

from mcp.server.fastmcp import FastMCP

# Contact fields never leave the server; recruiters get them from the ATS directly.
_PRIVATE_FIELDS = {"email", "phone"}


@cache
def _load(name: str) -> Any:
    return json.loads(resources.files("recruiting_agent.data").joinpath(name).read_text())


def jobs() -> list[dict]:
    return _load("jobs.json")


def candidates() -> list[dict]:
    return _load("candidates.json")


def policies() -> dict[str, str]:
    return _load("policies.json")


def _norm(values: list[str] | str | None) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = values.replace(",", " ").split()
    return [v.strip().lower() for v in values if v.strip()]


def _public(candidate: dict) -> dict:
    return {k: v for k, v in candidate.items() if k not in _PRIVATE_FIELDS}


def _summary(candidate: dict) -> dict:
    keys = ("id", "name", "headline", "years", "location", "skills")
    return {k: candidate[k] for k in keys}


# --- Pure functions (unit-testable without a transport) ---------------------------------


def search_jobs(query: str = "", location: str = "") -> dict:
    terms = _norm(query)
    loc = location.strip().lower()
    scored = []
    for job in jobs():
        if loc and loc not in job["location"].lower():
            continue
        haystack = " ".join([job["title"], job["description"], *job["skills"]]).lower()
        hits = sum(1 for t in terms if t in haystack)
        if terms and hits == 0:
            continue
        scored.append((hits, job))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return {"jobs": [job for _, job in scored]}


def get_job(job_id: str) -> dict:
    for job in jobs():
        if job["id"].lower() == job_id.strip().lower():
            return job
    return {"error": f"unknown job id {job_id}"}


def search_candidates(skills: list[str], min_years: int = 0, limit: int = 5) -> dict:
    wanted = set(_norm(skills))
    ranked = []
    for cand in candidates():
        if cand["years"] < min_years:
            continue
        overlap = len(wanted & set(cand["skills"]))
        if wanted and overlap == 0:
            continue
        ranked.append((overlap, cand["years"], cand))
    ranked.sort(key=lambda t: (-t[0], -t[1], t[2]["id"]))
    top = ranked[: max(1, limit)]
    return {"candidates": [{**_summary(c), "matched_skills": o} for o, _, c in top]}


def get_candidate(candidate_id: str) -> dict:
    for cand in candidates():
        if cand["id"].lower() == candidate_id.strip().lower():
            return _public(cand)
    return {"error": f"unknown candidate id {candidate_id}"}


def score_match(candidate_id: str, job_id: str) -> dict:
    cand, job = get_candidate(candidate_id), get_job(job_id)
    if "error" in cand or "error" in job:
        return {"error": cand.get("error") or job.get("error")}
    required = job["skills"]
    matched = [s for s in required if s in cand["skills"]]
    missing = [s for s in required if s not in cand["skills"]]
    coverage = len(matched) / len(required)
    experience = min(cand["years"] / job["min_years"], 1.0) if job["min_years"] else 1.0
    score = round(0.7 * coverage + 0.3 * experience, 2)
    return {
        "candidate_id": cand["id"],
        "job_id": job["id"],
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing,
        "years": cand["years"],
        "min_years": job["min_years"],
        "formula": "0.7 * skill_coverage + 0.3 * min(years / min_years, 1)",
        "note": "Decision support only; a recruiter makes the decision.",
    }


def get_policy(topic: str) -> dict:
    key = topic.strip().lower().replace(" ", "_")
    table = policies()
    if key not in table:
        return {"error": f"unknown topic {topic}", "topics": sorted(table)}
    return {"topic": key, "text": table[key]}


# --- MCP wiring --------------------------------------------------------------------------

mcp = FastMCP(
    "recruiting-data",
    instructions="Synthetic recruiting data: jobs, candidates, match scoring, hiring policy.",
)

mcp.tool(description="Search open jobs by keywords (skills or title words) and optional location.")(
    search_jobs
)
mcp.tool(description="Get one job posting by id, e.g. J-101.")(get_job)
mcp.tool(
    description="Rank candidates by overlap with the given skills. Never returns contact details."
)(search_candidates)
mcp.tool(description="Get one candidate profile by id, e.g. C-201. Contact details are excluded.")(
    get_candidate
)
mcp.tool(description="Transparent skill/experience match score between a candidate and a job.")(
    score_match
)
mcp.tool(description="Get hiring policy text. Topics: eeo, screening, data_privacy, ai_use.")(
    get_policy
)


@mcp.resource("policy://{topic}")
def policy_resource(topic: str) -> str:
    return get_policy(topic).get("text", "unknown topic")


@mcp.prompt()
def shortlist(job_id: str) -> str:
    """Prompt template: build a skills-only shortlist for a job."""
    return (
        f"Use get_job for {job_id}, then search_candidates with its skills and min_years. "
        "Return a ranked shortlist using only job-related evidence and cite every id."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true", help="serve streamable HTTP on :8765")
    args = parser.parse_args()
    if args.http:
        mcp.settings.host, mcp.settings.port = "0.0.0.0", 8765
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
