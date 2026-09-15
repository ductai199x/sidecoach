from sidecoach.events import Frame, MovePlayed, Region

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
AFTER_E4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"


def test_move_played_roundtrips_through_json() -> None:
    event = MovePlayed(
        game_id="g1",
        ply=1,
        san="e4",
        uci="e2e4",
        fen_before=START,
        fen_after=AFTER_E4,
        by_user=True,
        think_ms=1200,
    )
    assert MovePlayed.model_validate_json(event.model_dump_json()) == event


def test_frame_region_roundtrips_as_named_tuple() -> None:
    frame = Frame(frame_id=7, monitor=0, region=Region(10, 20, 640, 640))
    restored = Frame.model_validate_json(frame.model_dump_json())
    assert restored.region == Region(10, 20, 640, 640)
    assert restored.region.w == 640
