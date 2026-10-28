from liquitun3s.cybersand import wrap_smart_link_response


def test_smart_link_wrapper_copies_authority_from_request():
    request = {
        "version": 1,
        "request_id": "obs-12:3",
        "run_id": "run-a",
        "actor_id": 2,
        "epoch": 7,
        "observation_id": "obs-12",
        "observed_tick": 1200,
        "expires_tick": 1320,
        "observation": {"tick": 1200},
    }
    response = wrap_smart_link_response(
        request,
        {
            "mode": "move_to",
            "goal": [400, 250],
            "speed": 0.75,
            "engage": False,
        },
        valid_for_ticks=30,
    )
    assert response["request_id"] == request["request_id"]
    assert response["epoch"] == request["epoch"]
    assert response["observed_tick"] == request["observed_tick"]
    assert response["intent"]["mode"] == "move_to"


def test_model_intent_cannot_override_request_identity():
    request = {
        "version": 1,
        "request_id": "authoritative",
        "run_id": "run-a",
        "actor_id": 0,
        "epoch": 1,
        "observation_id": "obs",
        "observed_tick": 10,
        "expires_tick": 130,
        "observation": {},
    }
    response = wrap_smart_link_response(
        request,
        {"mode": "hold", "goal": [0, 0], "speed": 0, "engage": False},
    )
    assert response["request_id"] == "authoritative"
