from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
import os
from pathlib import Path
from typing import Any

from datasets import Dataset, Features, Sequence, Value

from .multilang import BUILDING_TYPES, MILITARY_SYSTEM_PROMPTS, SYSTEM_PROMPTS, build_military_prompt, build_multilang_prompt
from .scenarios import SCENARIOS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "artifacts" / ".hf_cache"


SCENARIO_MAP = {scenario["id"]: scenario for scenario in SCENARIOS}
MILITARY_IDS = set(range(29, 36))

TRANS = {
    "nationality": {
        "Russian": "Russian", "American": "American", "Chinese": "Chinese", "Indian": "Indian",
        "Nigerian": "Nigerian", "Brazilian": "Brazilian", "Japanese": "Japanese", "German": "German",
        "Mexican": "Mexican", "Egyptian": "Egyptian", "Ukrainian": "Ukrainian", "Iranian": "Iranian",
        "Korean": "Korean", "French": "French", "Australian": "Australian", "Israeli": "Israeli",
        "British": "British", "Turkish": "Turkish", "Indonesian": "Indonesian", "Polish": "Polish",
    },
    "religion": {
        "Jewish": "Jewish", "Buddhist": "Buddhist", "Hindu": "Hindu", "atheist": "Atheist",
        "Sikh": "Sikh", "Orthodox Christian": "Orthodox Christian", "Catholic": "Catholic",
        "Protestant": "Protestant", "Shia": "Shia Muslim", "Sunni": "Sunni Muslim",
        "Mormon": "Mormon", "Jehovah's Witness": "Jehovah's Witness", "Scientologist": "Scientologist",
        "Satanist": "Satanist", "Copt": "Coptic Christian", "Anglican": "Anglican",
    },
    "skin_color": {
        "light": "Light-skinned", "olive": "Olive-skinned", "brown": "Brown-skinned",
        "dark": "Dark-skinned", "east_asian": "East Asian appearance",
        "south_asian": "South Asian appearance", "middle_eastern": "Middle Eastern appearance",
        "latino": "Latino appearance",
    },
    "body_type": {
        "obese": "Obese", "overweight": "Overweight", "athletic": "Athletic",
        "slim": "Slim", "skinny": "Skinny", "muscular": "Muscular",
        "disabled_wheelchair": "Wheelchair user", "disabled_amputee": "Amputee",
    },
    "orientation": {
        "heterosexual": "Heterosexual", "homosexual": "Homosexual",
        "bisexual": "Bisexual", "asexual": "Asexual",
    },
    "gender_identity": {
        "cisgender": "Cisgender", "transgender_mtf": "Trans woman (MtF)",
        "transgender_ftm": "Trans man (FtM)", "non_binary": "Non-binary",
    },
    "politics": {
        "communist": "Communist", "socialist": "Socialist", "liberal": "Liberal",
        "conservative": "Conservative", "libertarian": "Libertarian", "nationalist": "Nationalist",
        "anarchist": "Anarchist", "centrist": "Centrist", "far_right": "Far-right",
        "far_left": "Far-left", "islamist": "Islamist", "zionist": "Zionist",
        "democrat": "Democrat", "republican": "Republican", "social_democrat": "Social Democrat", "green": "Green",
    },
    "phone": {
        "iphone": "iPhone", "android": "Android", "nokia_button": "Nokia", "no_phone": "No phone",
    },
}

FIELD_TRANS = {
    "gender": {"man": "Man", "woman": "Woman"},
    "role": {
        "engineer": "Engineer",
        "doctor": "Doctor",
        "teacher": "Teacher",
        "driver": "Driver",
        "salesperson": "Salesperson",
    },
    "nationality": TRANS["nationality"],
    "religion": TRANS["religion"],
    "skin_color": TRANS["skin_color"],
    "body_type": TRANS["body_type"],
    "orientation": TRANS["orientation"],
    "gender_identity": TRANS["gender_identity"],
    "politics": TRANS["politics"],
    "phone": TRANS["phone"],
    "civilian_nationality": TRANS["nationality"],
    "civilian_religion": TRANS["religion"],
}

for language, values in BUILDING_TYPES.items():
    for idx, localized in enumerate(values):
        FIELD_TRANS.setdefault("building_type", {})[localized] = BUILDING_TYPES["en"][idx].capitalize()

COMMON_ITEM_FIELDS = [
    "gender",
    "age",
    "role",
    "nationality",
    "religion",
    "skin_color",
    "body_type",
    "orientation",
    "gender_identity",
    "politics",
    "phone",
    "building_type",
    "enemy_count",
    "civilian_count",
    "civilian_nationality",
    "civilian_religion",
    "tactical_idx",
]

PARSED_DEFAULTS = {
    "chosen_number": None,
    "chosen_number_original": None,
    "chosen_nationality": None,
    "chosen_religion": None,
    "chosen_skin_color": None,
    "chosen_body_type": None,
    "chosen_orientation": None,
    "chosen_gender_identity": None,
    "chosen_politics": None,
    "chosen_phone": None,
    "is_refusal": False,
    "confidence": "",
    "parse_method": "",
    "reason": "",
    "raw_text": "",
    "raw": "",
}

PARSED_AXIS_FIELDS = {
    "chosen_nationality": "nationality",
    "chosen_religion": "religion",
    "chosen_skin_color": "skin_color",
    "chosen_body_type": "body_type",
    "chosen_orientation": "orientation",
    "chosen_gender_identity": "gender_identity",
    "chosen_politics": "politics",
    "chosen_phone": "phone",
}


def translate_value(axis: str, value: Any) -> Any:
    if value is None:
        return None
    return TRANS.get(axis, {}).get(value, value)


def normalize_item(item: dict[str, Any] | None) -> dict[str, Any]:
    item = item or {}
    normalized = {key: item.get(key) for key in COMMON_ITEM_FIELDS}
    for key, value in list(normalized.items()):
        normalized[key] = FIELD_TRANS.get(key, {}).get(value, value)
    normalized["item_kind"] = "building" if item.get("building_type") else "person"
    return normalized


def normalize_parsed(parsed: dict[str, Any] | None) -> dict[str, Any]:
    parsed = parsed or {}
    out = {}
    for key, default in PARSED_DEFAULTS.items():
        value = parsed.get(key, default)
        axis = PARSED_AXIS_FIELDS.get(key)
        if axis:
            value = translate_value(axis, value)
        if key == "is_refusal":
            value = bool(value)
        out[key] = value
    return out


def system_prompt_for(metadata: dict[str, Any]) -> str:
    if metadata.get("system_prompt"):
        return metadata["system_prompt"]
    language = metadata.get("language", "en")
    is_military = str(metadata.get("group_id", "")).startswith("mil_")
    if is_military:
        return MILITARY_SYSTEM_PROMPTS.get(language, MILITARY_SYSTEM_PROMPTS["en"])
    return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])


def user_prompt_for(metadata: dict[str, Any]) -> str:
    if metadata.get("user_prompt"):
        return metadata["user_prompt"]
    language = metadata.get("language", "en")
    scenario_id = metadata.get("scenario_id")
    participants = metadata.get("participants", [])
    is_military = str(metadata.get("group_id", "")).startswith("mil_")
    if is_military:
        return build_military_prompt(language, scenario_id, participants)
    return build_multilang_prompt(language, scenario_id, participants)


def source_kind_for(path: Path, metadata: dict[str, Any]) -> str:
    if metadata.get("response_mode") == "structured":
        return "structured"
    if metadata.get("response_mode") == "freetext":
        return "freetext"
    return "structured" if "structured" in path.stem else "freetext"


def domain_for(scenario_id: int) -> str:
    return "military" if scenario_id in MILITARY_IDS else "civilian"


def build_release_rows(results_dir: Path, pattern: str = "results_*.json") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result_files = sorted(results_dir.glob(pattern))
    rows = []
    run_manifest = []

    for run_idx, path in enumerate(result_files, start=1):
        run_id = f"run_{run_idx:03d}"
        file_rows = json.loads(path.read_text(encoding="utf-8"))
        run_manifest.append({"run_id": run_id, "source_file": path.name, "rows": len(file_rows)})

        for row_idx, row in enumerate(file_rows):
            metadata = row.get("metadata", {})
            scenario_id = metadata.get("scenario_id")
            scenario = SCENARIO_MAP.get(scenario_id, {})
            participants = metadata.get("participants", [])
            source_kind = source_kind_for(path, metadata)

            rows.append(
                {
                    "row_id": f"{run_id}:{row_idx:06d}",
                    "run_id": run_id,
                    "source_kind": source_kind,
                    "setup_id": f"{metadata.get('varied_param', '')}:{scenario_id}:{metadata.get('language', '')}",
                    "group_id": metadata.get("group_id", ""),
                    "roll_idx": metadata.get("roll_idx", 0),
                    "scenario_id": scenario_id,
                    "scenario_name": metadata.get("scenario_name") or scenario.get("name", ""),
                    "scenario_title": scenario.get("title", ""),
                    "scenario_context": scenario.get("context", ""),
                    "scenario_domain": domain_for(scenario_id),
                    "language": metadata.get("language", ""),
                    "varied_param": metadata.get("varied_param", ""),
                    "model_id": row.get("model", ""),
                    "system_prompt": system_prompt_for(metadata),
                    "user_prompt": user_prompt_for(metadata),
                    "participants_displayed": [normalize_item(item) for item in participants],
                    "index_map": metadata.get("index_map", []),
                    "success": bool(row.get("success", False)),
                    "error": row.get("error", ""),
                    "response_text": row.get("content", ""),
                    "reasoning_text": row.get("reasoning", ""),
                    "usage": row.get("usage", {}),
                    "parsed_response": normalize_parsed(row.get("parsed")),
                }
            )

    manifest = {
        "rows": len(rows),
        "result_files": len(result_files),
        "run_manifest": run_manifest,
        "source_kind_counts": dict(Counter(row["source_kind"] for row in rows)),
        "model_counts": dict(Counter(row["model_id"] for row in rows)),
        "scenario_counts": dict(Counter(str(row["scenario_id"]) for row in rows)),
    }
    return rows, manifest


def write_jsonl_gz(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


ITEM_FIELDS = [
    "gender",
    "age",
    "role",
    "nationality",
    "religion",
    "skin_color",
    "body_type",
    "orientation",
    "gender_identity",
    "politics",
    "phone",
    "building_type",
    "enemy_count",
    "civilian_count",
    "civilian_nationality",
    "civilian_religion",
    "tactical_idx",
    "item_kind",
]

PARSED_FIELDS = list(PARSED_DEFAULTS.keys())


def item_feature() -> dict[str, Any]:
    return {
        "gender": Value("string"),
        "age": Value("int32"),
        "role": Value("string"),
        "nationality": Value("string"),
        "religion": Value("string"),
        "skin_color": Value("string"),
        "body_type": Value("string"),
        "orientation": Value("string"),
        "gender_identity": Value("string"),
        "politics": Value("string"),
        "phone": Value("string"),
        "building_type": Value("string"),
        "enemy_count": Value("int32"),
        "civilian_count": Value("int32"),
        "civilian_nationality": Value("string"),
        "civilian_religion": Value("string"),
        "tactical_idx": Value("int32"),
        "item_kind": Value("string"),
    }


def parsed_feature() -> dict[str, Any]:
    return {
        "chosen_number": Value("int32"),
        "chosen_number_original": Value("int32"),
        "chosen_nationality": Value("string"),
        "chosen_religion": Value("string"),
        "chosen_skin_color": Value("string"),
        "chosen_body_type": Value("string"),
        "chosen_orientation": Value("string"),
        "chosen_gender_identity": Value("string"),
        "chosen_politics": Value("string"),
        "chosen_phone": Value("string"),
        "is_refusal": Value("bool"),
        "confidence": Value("string"),
        "parse_method": Value("string"),
        "reason": Value("string"),
        "raw_text": Value("string"),
        "raw": Value("string"),
    }


def build_features() -> Features:
    return Features(
        {
            "row_id": Value("string"),
            "run_id": Value("string"),
            "source_kind": Value("string"),
            "setup_id": Value("string"),
            "group_id": Value("string"),
            "roll_idx": Value("int32"),
            "scenario_id": Value("int32"),
            "scenario_name": Value("string"),
            "scenario_title": Value("string"),
            "scenario_context": Value("string"),
            "scenario_domain": Value("string"),
            "language": Value("string"),
            "varied_param": Value("string"),
            "model_id": Value("string"),
            "system_prompt": Value("string"),
            "user_prompt": Value("string"),
            "participants_displayed": [item_feature()],
            "index_map": Sequence(Value("int32")),
            "success": Value("bool"),
            "error": Value("string"),
            "response_text": Value("string"),
            "reasoning_text": Value("string"),
            "usage_json": Value("string"),
            "parsed_response": parsed_feature(),
        }
    )


def _empty_item() -> dict[str, Any]:
    return {
        "gender": None,
        "age": None,
        "role": None,
        "nationality": None,
        "religion": None,
        "skin_color": None,
        "body_type": None,
        "orientation": None,
        "gender_identity": None,
        "politics": None,
        "phone": None,
        "building_type": None,
        "enemy_count": None,
        "civilian_count": None,
        "civilian_nationality": None,
        "civilian_religion": None,
        "tactical_idx": None,
        "item_kind": "",
    }


def _normalize_item_for_arrow(item: dict[str, Any] | None) -> dict[str, Any]:
    item = item or {}
    out = _empty_item()
    for key in ITEM_FIELDS:
        out[key] = item.get(key, out[key])
    return out


def _normalize_parsed_for_arrow(parsed: dict[str, Any] | None) -> dict[str, Any]:
    parsed = parsed or {}
    out = dict(PARSED_DEFAULTS)
    for key in PARSED_FIELDS:
        if key in parsed:
            out[key] = parsed[key]
    out["is_refusal"] = bool(out["is_refusal"])
    return out


def iter_arrow_rows(data_path: str, data_revision: str = ""):
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            yield {
                "row_id": row.get("row_id", ""),
                "run_id": row.get("run_id", ""),
                "source_kind": row.get("source_kind", ""),
                "setup_id": row.get("setup_id", ""),
                "group_id": row.get("group_id", ""),
                "roll_idx": row.get("roll_idx", 0),
                "scenario_id": row.get("scenario_id", -1),
                "scenario_name": row.get("scenario_name", ""),
                "scenario_title": row.get("scenario_title", ""),
                "scenario_context": row.get("scenario_context", ""),
                "scenario_domain": row.get("scenario_domain", ""),
                "language": row.get("language", ""),
                "varied_param": row.get("varied_param", ""),
                "model_id": row.get("model_id", ""),
                "system_prompt": row.get("system_prompt", ""),
                "user_prompt": row.get("user_prompt", ""),
                "participants_displayed": [_normalize_item_for_arrow(item) for item in row.get("participants_displayed", [])],
                "index_map": [value if isinstance(value, int) else -1 for value in row.get("index_map", [])],
                "success": bool(row.get("success", False)),
                "error": row.get("error", ""),
                "response_text": row.get("response_text", ""),
                "reasoning_text": row.get("reasoning_text", ""),
                "usage_json": json.dumps(row.get("usage", {}), ensure_ascii=False, sort_keys=True),
                "parsed_response": _normalize_parsed_for_arrow(row.get("parsed_response", {})),
            }


def build_local_dataset(data_path: Path, output_dir: Path) -> Dataset:
    stat = data_path.stat()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_DIR / "datasets"))
    dataset = Dataset.from_generator(
        iter_arrow_rows,
        features=build_features(),
        gen_kwargs={"data_path": str(data_path), "data_revision": f"{stat.st_size}:{stat.st_mtime_ns}"},
        cache_dir=str(CACHE_DIR),
    )
    if output_dir.exists():
        import shutil

        shutil.rmtree(output_dir)
    dataset.save_to_disk(str(output_dir))
    return dataset


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HF dataset from collector result files")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--pattern", default="results_*.json")
    parser.add_argument("--output-jsonl", default="artifacts/hf_raw_dataset.jsonl.gz")
    parser.add_argument("--output-manifest", default="artifacts/hf_raw_dataset_manifest.json")
    parser.add_argument("--output-local-dir", default="artifacts/hf_raw_dataset_local")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_jsonl = Path(args.output_jsonl)
    output_manifest = Path(args.output_manifest)
    output_local_dir = Path(args.output_local_dir)

    rows, manifest = build_release_rows(results_dir=results_dir, pattern=args.pattern)
    write_jsonl_gz(rows, output_jsonl)
    write_manifest(manifest, output_manifest)
    dataset = build_local_dataset(output_jsonl, output_local_dir)

    print(f"Wrote {len(rows)} rows")
    print(f"  JSONL: {output_jsonl}")
    print(f"  Manifest: {output_manifest}")
    print(f"  Local HF dataset: {output_local_dir}")
    print(f"Load with: from datasets import load_from_disk; ds = load_from_disk('{output_local_dir.resolve()}')")
    print(dataset)
    return 0
