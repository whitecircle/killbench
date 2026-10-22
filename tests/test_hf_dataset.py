import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from killbench_collector.hf_dataset import parsed_feature, PARSED_DEFAULTS, PARSED_AXIS_FIELDS


def test_parsed_feature_schema_includes_all_fields():
    """Test that all fields from PARSED_DEFAULTS are included in the schema."""
    schema = parsed_feature()
    for field in PARSED_DEFAULTS.keys():
        assert field in schema, f"Field {field} from PARSED_DEFAULTS missing in schema"


def test_chosen_gender_identity_in_schema():
    """Regression test: verify chosen_gender_identity is in the output schema."""
    schema = parsed_feature()
    assert "chosen_gender_identity" in schema, "chosen_gender_identity missing from schema"
    assert "string" in str(schema["chosen_gender_identity"]).lower(), \
        "chosen_gender_identity should be string type"


def test_parsed_feature_includes_all_axis_fields():
    """Test that all fields from PARSED_AXIS_FIELDS are in the schema."""
    schema = parsed_feature()
    for field in PARSED_AXIS_FIELDS.keys():
        assert field in schema, f"Axis field {field} missing from schema"
