import repair_attainment_campaign as campaign


def test_transition_counts_are_exhaustive_with_destination_precedence():
    def move(session, method, destination):
        return {"session_id": session, "method": method,
                "destination_instance": destination}

    row = {
        "initial_moves": [
            move("locked", "replay", "east"),
            move("same", "replay", "east"),
            move("method", "replay", "east"),
            move("destination", "replay", "east"),
            move("removed", "replay", "east"),
        ],
        "repair_moves": [
            move("locked", "replay", "east"),
            move("same", "replay", "east"),
            move("method", "kv_transfer", "east"),
            move("destination", "kv_transfer", "germany"),
            move("added", "replay", "germany"),
        ],
        "repair_schedule": [{"session_id": "locked",
                             "status": "committed_before_event"}],
    }
    assert campaign.transition_counts(row) == {
        "pending": 4, "retained": 1, "method": 1,
        "destination": 1, "removed": 1,
    }
