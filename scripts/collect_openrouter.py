#!/usr/bin/env python3
"""Clean OpenRouter collector for Eat-a-Kid Bench.

This collector keeps the existing prompt/test generation logic, but uses:
- one OpenRouter client path for inference
- canonical Gemini parsing for free-text responses
- one CLI with a small-sample debug path

Examples:
    python3 collect_openrouter.py --response-mode freetext --mode nationality --model openai/gpt-5.4 --limit 4 --dry-run
    python3 collect_openrouter.py --response-mode freetext --mode nationality --model openai/gpt-5.4 --limit 4
    python3 collect_openrouter.py --response-mode structured --mode religion --model google/gemini-3.1-pro-preview --limit 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from killbench_collector.config import BENCH_MODELS, BENCH_REASONING, CONCURRENCY_LIMIT
from killbench_collector.gemini_parser import GeminiFreeTextParser
from killbench_collector.multilang import MILITARY_SYSTEM_PROMPTS, SYSTEM_PROMPTS
from killbench_collector.openrouter_client import OpenRouterClient
from killbench_collector.results import RESULTS_DIR, save_results
from killbench_collector.scenarios import SCENARIOS
from killbench_collector.test_generation import ALL_MODES, generate_tests_for_mode, map_choice_to_original


CHOSEN_FIELDS = [
    "nationality",
    "religion",
    "skin_color",
    "phone",
    "body_type",
    "orientation",
    "gender_identity",
    "politics",
]

SCENARIO_MAP = {scenario["id"]: scenario for scenario in SCENARIOS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean OpenRouter collector")
    parser.add_argument(
        "--response-mode",
        choices=["freetext", "structured"],
        required=True,
        help="Free-text collection with Gemini parsing, or structured JSON collection.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        dest="modes",
        choices=["all"] + ALL_MODES,
        required=True,
        help="Benchmark mode to run. Repeat the flag for multiple modes.",
    )
    parser.add_argument(
        "--language",
        action="append",
        dest="languages",
        help="Language code to run. Repeat for multiple languages. Defaults to all languages.",
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        dest="scenario_ids",
        type=int,
        help="Scenario id to run. Repeat for multiple scenarios. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model id to run. Repeat for multiple models. Defaults to all configured models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the total number of API jobs after expansion across modes and models.",
    )
    parser.add_argument(
        "--per-scenario-limit",
        type=int,
        default=0,
        help="Limit jobs per (model, source mode, scenario) cell after expansion.",
    )
    parser.add_argument(
        "--split-per-language",
        action="store_true",
        help="When used with --per-scenario-limit, apply the limit separately for each language.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="If set with --limit, randomly sample jobs instead of taking the first N.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=CONCURRENCY_LIMIT,
        help="OpenRouter concurrency limit for both inference and Gemini parsing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without making API calls.")
    parser.add_argument(
        "--skip-gemini-parse",
        action="store_true",
        help="Free-text only: skip the Gemini parser overlay and keep the raw parser fields empty.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120).",
    )
    return parser.parse_args()


def expand_modes(requested_modes: list[str]) -> list[str]:
    if "all" in requested_modes:
        return list(ALL_MODES)
    seen = []
    for mode in requested_modes:
        if mode not in seen:
            seen.append(mode)
    return seen


def select_models(requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return list(BENCH_MODELS)


def build_jobs(modes: list[str], models: list[str]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for mode in modes:
        tests = generate_tests_for_mode(mode)
        for group in tests:
            for model in models:
                is_reasoning = model in BENCH_REASONING
                system_prompt = system_prompt_for(group)
                scenario = SCENARIO_MAP.get(group["scenario_id"], {})
                metadata = {
                    **{k: v for k, v in group.items() if k != "prompt"},
                    "source_mode": mode,
                    "is_reasoning": is_reasoning,
                    "system_prompt": system_prompt,
                    "user_prompt": group["prompt"],
                    "scenario_name": scenario.get("name", ""),
                }
                jobs.append(
                    {
                        "model": model,
                        "prompt": group["prompt"],
                        "metadata": metadata,
                        "is_reasoning": is_reasoning,
                        "system_prompt": system_prompt,
                    }
                )
    return jobs


def filter_jobs_by_language(jobs: list[dict[str, Any]], languages: list[str] | None) -> list[dict[str, Any]]:
    if not languages:
        return jobs
    wanted = set(languages)
    return [job for job in jobs if job["metadata"].get("language") in wanted]


def filter_jobs_by_scenario(jobs: list[dict[str, Any]], scenario_ids: list[int] | None) -> list[dict[str, Any]]:
    if not scenario_ids:
        return jobs
    wanted = set(scenario_ids)
    return [job for job in jobs if int(job["metadata"].get("scenario_id", 0)) in wanted]


def limit_jobs(jobs: list[dict[str, Any]], limit: int, sample_seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(jobs):
        return jobs
    if sample_seed:
        rng = random.Random(sample_seed)
        return rng.sample(jobs, limit)
    return jobs[:limit]


def limit_jobs_per_scenario(
    jobs: list[dict[str, Any]],
    per_scenario_limit: int,
    sample_seed: int,
    split_per_language: bool,
) -> list[dict[str, Any]]:
    if per_scenario_limit <= 0:
        return jobs

    grouped: dict[tuple[str, str, int, str], list[tuple[int, dict[str, Any]]]] = {}
    for idx, job in enumerate(jobs):
        metadata = job["metadata"]
        key = (
            job["model"],
            str(metadata.get("source_mode", metadata.get("varied_param", ""))),
            int(metadata.get("scenario_id", 0)),
            str(metadata.get("language", "")) if split_per_language else "",
        )
        grouped.setdefault(key, []).append((idx, job))

    rng = random.Random(sample_seed) if sample_seed else None
    selected: list[tuple[int, dict[str, Any]]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        if len(rows) <= per_scenario_limit:
            chosen = rows
        elif rng is not None:
            chosen = rng.sample(rows, per_scenario_limit)
        else:
            chosen = rows[:per_scenario_limit]
        selected.extend(chosen)

    selected.sort(key=lambda pair: pair[0])
    return [job for _, job in selected]


def system_prompt_for(metadata: dict[str, Any]) -> str:
    lang = metadata.get("language", "en")
    is_military = str(metadata.get("group_id", "")).startswith("mil_")
    if is_military:
        return MILITARY_SYSTEM_PROMPTS.get(lang, MILITARY_SYSTEM_PROMPTS["en"])
    return SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])


def parse_structured_content(content: str) -> dict[str, Any]:
    import re

    if not content or not content.strip():
        return {
            "chosen_number": None,
            "is_refusal": True,
            "parse_method": "structured_empty",
            "reason": "",
        }

    stripped = content.strip()
    if not stripped.startswith("{"):
        import re

        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1)

    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        partial_match = re.search(r'"chosen_number"\s*:\s*([1-4])', stripped)
        if partial_match:
            return {
                "chosen_number": int(partial_match.group(1)),
                "is_refusal": False,
                "parse_method": "structured_partial_json",
                "reason": "",
                "raw": stripped[:200],
            }
        return {
            "chosen_number": None,
            "is_refusal": True,
            "parse_method": "structured_parse_error",
            "reason": "",
            "raw": stripped[:200],
        }

    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        data = {}

    chosen = data.get("chosen_number")
    reason = data.get("reason", "")
    if isinstance(chosen, int) and 1 <= chosen <= 4:
        return {
            "chosen_number": chosen,
            "is_refusal": False,
            "parse_method": "structured",
            "reason": reason,
        }
    return {
        "chosen_number": None,
        "is_refusal": True,
        "parse_method": "structured_invalid",
        "reason": reason,
    }


def empty_parsed(parse_method: str, raw_text: str = "", is_refusal: bool = False) -> dict[str, Any]:
    parsed = {
        "chosen_number": None,
        "chosen_number_original": None,
        "is_refusal": is_refusal,
        "confidence": "none",
        "parse_method": parse_method,
        "raw_text": raw_text,
    }
    for field in CHOSEN_FIELDS:
        parsed[f"chosen_{field}"] = None
    return parsed


def enrich_with_original_choice(parsed: dict[str, Any], metadata: dict[str, Any], chosen_displayed: int | None) -> dict[str, Any]:
    out = dict(parsed)
    out["chosen_number"] = chosen_displayed
    original_choice = map_choice_to_original(chosen_displayed, metadata.get("index_map", []))
    out["chosen_number_original"] = original_choice
    original_participants = metadata.get("original_participants", metadata.get("participants", []))
    if isinstance(original_choice, int) and 1 <= original_choice <= len(original_participants):
        participant = original_participants[original_choice - 1]
        for field in CHOSEN_FIELDS:
            civil_key = f"civilian_{field}" if field in ("nationality", "religion") else field
            out[f"chosen_{field}"] = participant.get(field, participant.get(civil_key))
    return out


def build_free_text_fallback(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("success"):
        parsed = empty_parsed("request_error", result.get("error", ""), is_refusal=False)
        parsed["confidence"] = "none"
        return parsed
    if not (result.get("content") or "").strip():
        parsed = empty_parsed("no_response", "", is_refusal=False)
        parsed["confidence"] = "none"
        return parsed
    parsed = empty_parsed("gemini_reparse_skipped", result.get("content", ""), is_refusal=False)
    parsed["confidence"] = "low"
    return parsed


async def collect_jobs(
    jobs: list[dict[str, Any]],
    *,
    response_mode: str,
    concurrency: int,
    skip_gemini_parse: bool,
    incremental_path: Path | None = None,
    timeout_seconds: int = 120,
) -> list[dict[str, Any]]:
    # Split jobs by model for parallel pipelines
    jobs_by_model: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        jobs_by_model.setdefault(job["model"], []).append(job)

    n_models = len(jobs_by_model)
    # Each model gets its own TCP connector headroom
    connector = aiohttp.TCPConnector(limit=concurrency * n_models + 50)

    # Prepare incremental output dir
    inc_dir: Path | None = None
    if incremental_path:
        inc_dir = incremental_path.parent
        inc_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(total=len(jobs), desc=response_mode.upper())
    all_results: list[dict[str, Any]] = []
    results_lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector) as session:
        client = OpenRouterClient(session, concurrency=concurrency, timeout_seconds=timeout_seconds)

        async def run_model_pipeline(model: str, model_jobs: list[dict[str, Any]]) -> None:
            model_tag = model.replace("/", "_")
            inc_file = None
            if inc_dir:
                inc_path = inc_dir / f"incremental_{model_tag}.jsonl"
                inc_file = open(inc_path, "a", encoding="utf-8")

            async def run_one(job: dict[str, Any]) -> dict[str, Any]:
                response = await client.complete(
                    model=job["model"],
                    system_prompt=job["system_prompt"],
                    user_prompt=job["prompt"],
                    response_mode="structured" if response_mode == "structured" else "text",
                )
                response["metadata"] = job["metadata"]
                return response

            # Fire all jobs for this model concurrently (semaphore limits in client)
            tasks = [asyncio.create_task(run_one(job)) for job in model_jobs]

            for task in asyncio.as_completed(tasks):
                result = await task

                # Parse structured inline
                if response_mode == "structured":
                    if result.get("success"):
                        parsed = parse_structured_content(result.get("content", ""))
                        parsed = enrich_with_original_choice(parsed, result["metadata"], parsed.get("chosen_number"))
                    else:
                        parsed = empty_parsed("request_error", result.get("error", ""), is_refusal=False)
                    result["parsed"] = parsed

                elif skip_gemini_parse:
                    result["parsed"] = build_free_text_fallback(result)

                # Incremental flush per result
                if inc_file:
                    inc_file.write(json.dumps({
                        "model": result.get("model", ""),
                        "metadata": result.get("metadata", {}),
                        "parsed": result.get("parsed", {}),
                        "content": result.get("content", ""),
                        "success": result.get("success", False),
                        "error": result.get("error", ""),
                        "usage": result.get("usage", {}),
                    }, ensure_ascii=False) + "\n")
                    inc_file.flush()

                async with results_lock:
                    all_results.append(result)
                pbar.update(1)

            if inc_file:
                inc_file.close()

        # Run all model pipelines concurrently
        await asyncio.gather(*(
            run_model_pipeline(model, model_jobs)
            for model, model_jobs in jobs_by_model.items()
        ))

    pbar.close()

    # For freetext mode, run Gemini parsing pass
    if response_mode != "structured":
        gemini_parser = None
        parser_map: dict[int, dict[str, Any]] = {}
        if not skip_gemini_parse:
            connector2 = aiohttp.TCPConnector(limit=concurrency + 20)
            async with aiohttp.ClientSession(connector=connector2) as session2:
                client2 = OpenRouterClient(session2, concurrency=concurrency)
                gemini_parser = GeminiFreeTextParser(client2)

                parse_candidates = [
                    (idx, result)
                    for idx, result in enumerate(all_results)
                    if result.get("success") and (result.get("content") or "").strip()
                ]

                if parse_candidates:
                    ppbar = tqdm(total=len(parse_candidates), desc="GEMINI-PARSE")

                    async def run_parse(ridx: int, res: dict[str, Any], n_opt: int) -> tuple[int, dict[str, Any]]:
                        parsed = await gemini_parser.parse_response(res.get("content", ""), n_opt)
                        return ridx, parsed

                    ptasks = [
                        asyncio.create_task(run_parse(idx, result, len(result["metadata"].get("participants", [])) or 4))
                        for idx, result in parse_candidates
                    ]
                    for ptask in asyncio.as_completed(ptasks):
                        idx, parsed = await ptask
                        parser_map[idx] = parsed
                        ppbar.update(1)
                    ppbar.close()

        for idx, result in enumerate(all_results):
            if idx not in parser_map:
                result["parsed"] = build_free_text_fallback(result)
                continue
            parsed_result = parser_map[idx]
            if parsed_result.get("is_refusal"):
                parsed = empty_parsed("gemini_reparse", result.get("content", ""), is_refusal=True)
                parsed["confidence"] = "gemini_reparse"
                parsed["raw"] = parsed_result.get("raw", "")
            elif isinstance(parsed_result.get("displayed_choice"), int):
                parsed = empty_parsed("gemini_reparse", result.get("content", ""), is_refusal=False)
                parsed["confidence"] = "gemini_reparse"
                parsed["raw"] = parsed_result.get("raw", "")
                parsed = enrich_with_original_choice(parsed, result["metadata"], parsed_result["displayed_choice"])
            else:
                parsed = empty_parsed(parsed_result.get("parse_method", "gemini_reparse_error"), result.get("content", ""), is_refusal=False)
                parsed["confidence"] = "gemini_reparse"
                parsed["raw"] = parsed_result.get("raw", "")
                if parsed_result.get("error"):
                    parsed["error"] = parsed_result["error"]
            result["parsed"] = parsed

    return all_results


def print_summary(results: list[dict[str, Any]], response_mode: str) -> None:
    total = len(results)
    errors = sum(1 for r in results if not r.get("success"))
    refusals = sum(1 for r in results if r.get("parsed", {}).get("is_refusal"))
    choices = sum(1 for r in results if r.get("parsed", {}).get("chosen_number_original") is not None)
    print(f"\nSummary: {total} total | {choices} choices | {refusals} refusals | {errors} errors")

    print("\nPer-model summary:")
    by_model = {}
    for result in results:
        by_model.setdefault(result.get("model", ""), []).append(result)
    for model in sorted(by_model):
        rows = by_model[model]
        model_errors = sum(1 for r in rows if not r.get("success"))
        model_refusals = sum(1 for r in rows if r.get("parsed", {}).get("is_refusal"))
        model_choices = sum(1 for r in rows if r.get("parsed", {}).get("chosen_number_original") is not None)
        parse_methods = Counter(r.get("parsed", {}).get("parse_method", "?") for r in rows)
        print(
            f"  {model:45s} {model_choices:4d} choices | {model_refusals:4d} refusals | {model_errors:4d} errors | parse={dict(parse_methods)}"
        )

    if response_mode == "freetext":
        print("\nFree-text note: canonical parsed output comes from Gemini reparse.")


async def async_main(args: argparse.Namespace) -> int:
    modes = expand_modes(args.modes)
    models = select_models(args.models)
    jobs = build_jobs(modes, models)
    jobs = filter_jobs_by_language(jobs, args.languages)
    jobs = filter_jobs_by_scenario(jobs, args.scenario_ids)
    for job in jobs:
        job["metadata"]["response_mode"] = args.response_mode
    jobs = limit_jobs_per_scenario(
        jobs,
        args.per_scenario_limit,
        args.sample_seed,
        args.split_per_language,
    )
    jobs = limit_jobs(jobs, args.limit, args.sample_seed)

    print(f"Response mode: {args.response_mode}")
    print(f"Modes: {modes}")
    print(f"Models ({len(models)}):")
    for model in models:
        print(f"  {model}")
    if args.languages:
        print(f"Languages: {args.languages}")
    if args.scenario_ids:
        print(f"Scenario ids: {args.scenario_ids}")
    if args.per_scenario_limit:
        scope = "per (model, mode, scenario, language)" if args.split_per_language else "per (model, mode, scenario)"
        print(f"Per-scenario limit: {args.per_scenario_limit} {scope}")
    print(f"Jobs: {len(jobs)}")

    if jobs:
        sample = jobs[0]
        meta = sample["metadata"]
        print("\nSample job:")
        print(f"  model={sample['model']}")
        print(f"  source_mode={meta.get('source_mode')}")
        print(f"  varied_param={meta.get('varied_param')}")
        print(f"  scenario_id={meta.get('scenario_id')}")
        print(f"  group_id={meta.get('group_id')}")
        print(f"  language={meta.get('language')}")

    if args.dry_run:
        print("\nDry run only. No API calls were made.")
        return 0

    mode_tag = "+".join(sorted(modes))
    run_kind = "bench" if args.response_mode == "freetext" else "structured"
    run_id = f"clean_{run_kind}_{mode_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    incremental_path = RESULTS_DIR / f"incremental_{run_id}.jsonl"

    results = await collect_jobs(
        jobs,
        response_mode=args.response_mode,
        concurrency=args.concurrency,
        skip_gemini_parse=args.skip_gemini_parse,
        incremental_path=incremental_path,
        timeout_seconds=args.timeout,
    )
    paths = save_results(results, run_id=run_id)
    print("\nSaved results:")
    print(f"  CSV:  {paths['csv']}")
    print(f"  JSON: {paths['json']}")
    print(f"  Raw:  {paths['raw']}")
    print_summary(results, args.response_mode)
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
