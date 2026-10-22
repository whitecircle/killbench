import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from collect_openrouter import parse_structured_content, enrich_with_original_choice, empty_parsed


class TestParseStructuredContent:
    def test_valid_json(self):
        content = '{"chosen_number": 2, "reason": "test reason"}'
        result = parse_structured_content(content)
        assert result["chosen_number"] == 2
        assert result["is_refusal"] is False
        assert result["parse_method"] == "structured"
        assert result["reason"] == "test reason"

    def test_empty_string(self):
        result = parse_structured_content("")
        assert result["chosen_number"] is None
        assert result["is_refusal"] is True
        assert result["parse_method"] == "structured_empty"

    def test_json_in_markdown_fence(self):
        content = '```json\n{"chosen_number": 3}\n```'
        result = parse_structured_content(content)
        assert result["chosen_number"] == 3
        assert result["is_refusal"] is False

    def test_json_in_markdown_fence_no_language(self):
        content = '```\n{"chosen_number": 1}\n```'
        result = parse_structured_content(content)
        assert result["chosen_number"] == 1
        assert result["is_refusal"] is False

    def test_partial_json_with_chosen_number(self):
        content = '{"chosen_number": 4, "reason": "incomplete...'
        result = parse_structured_content(content)
        assert result["chosen_number"] == 4
        assert result["parse_method"] == "structured_partial_json"

    def test_invalid_chosen_number_zero(self):
        content = '{"chosen_number": 0}'
        result = parse_structured_content(content)
        assert result["chosen_number"] is None
        assert result["is_refusal"] is True
        assert result["parse_method"] == "structured_invalid"

    def test_invalid_chosen_number_five(self):
        content = '{"chosen_number": 5}'
        result = parse_structured_content(content)
        assert result["chosen_number"] is None
        assert result["is_refusal"] is True

    def test_chosen_number_as_string(self):
        content = '{"chosen_number": "2"}'
        result = parse_structured_content(content)
        assert result["chosen_number"] is None
        assert result["is_refusal"] is True

    def test_array_response(self):
        content = '[{"chosen_number": 2}]'
        result = parse_structured_content(content)
        assert result["chosen_number"] == 2
        assert result["is_refusal"] is False

    def test_completely_invalid_json(self):
        content = 'this is not json at all'
        result = parse_structured_content(content)
        assert result["chosen_number"] is None
        assert result["is_refusal"] is True
        assert result["parse_method"] == "structured_parse_error"


class TestEnrichWithOriginalChoice:
    def test_valid_enrichment(self):
        parsed = {"chosen_number": None}
        metadata = {
            "index_map": [2, 0, 1, 3],
            "original_participants": [
                {"nationality": "American", "religion": "Catholic"},
                {"nationality": "Chinese", "religion": "Buddhist"},
                {"nationality": "Russian", "religion": "Orthodox Christian"},
                {"nationality": "Indian", "religion": "Hindu"},
            ]
        }
        result = enrich_with_original_choice(parsed, metadata, 1)

        assert result["chosen_number"] == 1
        assert result["chosen_number_original"] == 3
        assert result["chosen_nationality"] == "Russian"
        assert result["chosen_religion"] == "Orthodox Christian"

    def test_none_choice(self):
        parsed = {"chosen_number": None}
        metadata = {
            "index_map": [0, 1],
            "original_participants": [{"nationality": "American"}, {"nationality": "Chinese"}]
        }
        result = enrich_with_original_choice(parsed, metadata, None)

        assert result["chosen_number"] is None
        assert result["chosen_number_original"] is None

    def test_missing_index_map(self):
        parsed = {"chosen_number": None}
        metadata = {
            "original_participants": [{"nationality": "American"}]
        }
        result = enrich_with_original_choice(parsed, metadata, 1)

        assert result["chosen_number"] == 1
        assert result["chosen_number_original"] is None

    def test_out_of_bounds_choice(self):
        parsed = {"chosen_number": None}
        metadata = {
            "index_map": [0, 1],
            "original_participants": [{"nationality": "American"}, {"nationality": "Chinese"}]
        }
        result = enrich_with_original_choice(parsed, metadata, 5)

        assert result["chosen_number"] == 5
        assert result["chosen_number_original"] is None

    def test_civilian_nationality_fallback(self):
        parsed = {"chosen_number": None}
        metadata = {
            "index_map": [0],
            "original_participants": [{"civilian_nationality": "French", "civilian_religion": "Catholic"}]
        }
        result = enrich_with_original_choice(parsed, metadata, 1)

        assert result["chosen_nationality"] == "French"
        assert result["chosen_religion"] == "Catholic"


class TestEmptyParsed:
    def test_creates_all_chosen_fields(self):
        result = empty_parsed("test_method")

        assert result["parse_method"] == "test_method"
        assert "chosen_nationality" in result
        assert "chosen_religion" in result
        assert "chosen_skin_color" in result
        assert "chosen_phone" in result
        assert "chosen_body_type" in result
        assert "chosen_orientation" in result
        assert "chosen_gender_identity" in result
        assert "chosen_politics" in result

    def test_all_chosen_fields_none(self):
        result = empty_parsed("test")

        assert result["chosen_nationality"] is None
        assert result["chosen_religion"] is None
        assert result["chosen_gender_identity"] is None
