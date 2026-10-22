import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from killbench_collector.test_generation import shuffle_participants, map_choice_to_original


class TestMapChoiceToOriginal:
    def test_valid_choice(self):
        index_map = [2, 0, 3, 1]
        assert map_choice_to_original(1, index_map) == 3
        assert map_choice_to_original(2, index_map) == 1
        assert map_choice_to_original(3, index_map) == 4
        assert map_choice_to_original(4, index_map) == 2

    def test_none_input(self):
        index_map = [2, 0, 3, 1]
        assert map_choice_to_original(None, index_map) is None

    def test_empty_index_map(self):
        assert map_choice_to_original(1, []) is None

    def test_out_of_bounds_negative(self):
        index_map = [2, 0, 3, 1]
        assert map_choice_to_original(0, index_map) is None
        assert map_choice_to_original(-1, index_map) is None

    def test_out_of_bounds_positive(self):
        index_map = [2, 0, 3, 1]
        assert map_choice_to_original(5, index_map) is None
        assert map_choice_to_original(100, index_map) is None


class TestShuffleParticipants:
    def test_deterministic_with_same_seed(self):
        participants = [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]
        rng1 = random.Random(42)
        rng2 = random.Random(42)

        shuffled1, map1 = shuffle_participants(participants.copy(), rng1)
        shuffled2, map2 = shuffle_participants(participants.copy(), rng2)

        assert shuffled1 == shuffled2, "Same seed should produce same shuffle"
        assert map1 == map2, "Same seed should produce same index map"

    def test_all_participants_present(self):
        participants = [{"id": i} for i in range(10)]
        rng = random.Random(123)

        shuffled, index_map = shuffle_participants(participants, rng)

        assert len(shuffled) == len(participants), "Shuffle should preserve count"
        shuffled_ids = {p["id"] for p in shuffled}
        original_ids = {p["id"] for p in participants}
        assert shuffled_ids == original_ids, "All participants should be present"

    def test_index_map_correctness(self):
        participants = [{"id": i} for i in range(5)]
        rng = random.Random(999)

        shuffled, index_map = shuffle_participants(participants, rng)

        for shuffled_idx, original_idx in enumerate(index_map):
            assert shuffled[shuffled_idx]["id"] == participants[original_idx]["id"], \
                f"Index map incorrect at position {shuffled_idx}"

    def test_single_participant(self):
        participants = [{"id": 1}]
        rng = random.Random(42)

        shuffled, index_map = shuffle_participants(participants, rng)

        assert shuffled == participants
        assert index_map == [0]

    def test_empty_participants(self):
        participants = []
        rng = random.Random(42)

        shuffled, index_map = shuffle_participants(participants, rng)

        assert shuffled == []
        assert index_map == []
