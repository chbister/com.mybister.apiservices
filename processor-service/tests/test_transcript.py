"""Tests fuer den ASR-/Diarization-Merge und den /v1/transcript-Endpunkt.

Deterministisch, ohne Modelle oder externe Services.
Ausfuehren aus dem processor-service-Verzeichnis:  pytest tests/
"""

from fastapi.testclient import TestClient

from app.main import app
from app.transcript import (
    assign_speaker,
    build_transcript,
    format_transcript,
    group_utterances,
    merge_segments,
    split_interval,
)

client = TestClient(app)

TURNS = [
    {"start": 0.0, "end": 10.0, "speaker": "SPEAKER_00"},
    {"start": 10.0, "end": 20.0, "speaker": "SPEAKER_01"},
]

SEGMENTS = [
    {"start": 0.0, "end": 6.0, "text": "Hallo, wie geht es dir?"},
    {"start": 6.0, "end": 14.0, "text": "Gut, und selbst? Alles klar bei euch."},
    {"start": 14.0, "end": 20.0, "text": "Ja, super."},
]


def test_assign_speaker_max_overlap():
    assert assign_speaker(0.0, 6.0, TURNS) == "SPEAKER_00"
    assert assign_speaker(12.0, 18.0, TURNS) == "SPEAKER_01"
    # Kein Turn in Sicht -> ohne Turns bleibt es None
    assert assign_speaker(30.0, 35.0, []) is None
    # Intervall in der Pause (10.0) -> naechster Turn (Grenze)
    assert assign_speaker(9.5, 10.5, TURNS) in ("SPEAKER_00", "SPEAKER_01")
    # Nach dem letzten Turn -> naechster Turn (SPEAKER_01)
    assert assign_speaker(30.0, 35.0, TURNS) == "SPEAKER_01"


def test_split_interval_at_speaker_change():
    spans = split_interval(6.0, 14.0, TURNS)
    assert [(round(a, 1), round(b, 1), s) for a, b, s in spans] == [
        (6.0, 10.0, "SPEAKER_00"),
        (10.0, 14.0, "SPEAKER_01"),
    ]


def test_merge_splits_text_proportionally():
    merged = merge_segments([SEGMENTS[1]], TURNS)
    assert len(merged) == 2
    assert merged[0]["speaker"] == "SPEAKER_00"
    assert merged[1]["speaker"] == "SPEAKER_01"
    assert merged[0]["text"] + " " + merged[1]["text"] == SEGMENTS[1]["text"]
    assert merged[0]["end"] == 10.0
    assert merged[1]["start"] == 10.0


def test_group_utterances_merges_consecutive_same_speaker():
    merged = merge_segments(SEGMENTS, TURNS)
    utterances = group_utterances(merged)
    # Segment 1 komplett SPEAKER_00, Segment 2 wird an 10.0 gesplittet
    # (00|01), Segment 3 komplett SPEAKER_01 -> 00 | 01
    speakers = [u["speaker"] for u in utterances]
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]
    # 6-10 (00) + 10-14 (01) + 14-20 (01) -> 01 haelt 10-20
    assert utterances[1]["start"] == 10.0
    assert utterances[1]["end"] == 20.0


def test_format_transcript_timestamps_and_names():
    merged = merge_segments(SEGMENTS, TURNS)
    utterances = group_utterances(merged)
    names = {"SPEAKER_00": "Anna", "SPEAKER_01": "Sprecher 2"}
    text = format_transcript(utterances, names)
    lines = text.splitlines()
    # Segment 2 wird an der Sprecherwechsel-Grenze (10.0) an Wortgrenzen
    # gesplittet und mit Segment 1 zum Aussen-Block gruppiert.
    assert lines[0] == (
        "[00:00:00] Anna: Hallo, wie geht es dir? Gut, und selbst? Alles"
    )
    assert lines[1] == "[00:00:10] Sprecher 2: klar bei euch. Ja, super."


def test_build_transcript_speaker_summary_and_counts():
    result = build_transcript(SEGMENTS, TURNS, language="de")
    assert result["language"] == "de"
    speakers = {s["speaker"]: s for s in result["speakers"]}
    assert speakers["SPEAKER_00"]["display_name"] == "Sprecher 1"
    assert speakers["SPEAKER_01"]["first_appearance"] == 10.0
    assert result["counts"]["turns"] == 2
    assert result["transcript"].count("\n") == 1  # 2 Aeusserungen -> 2 Zeilen


def test_build_transcript_without_diarization():
    result = build_transcript(SEGMENTS, [])
    assert all(seg["speaker"] is None for seg in result["segments"])
    assert "Unbekannt" in result["transcript"]


def test_endpoint_flat_lists():
    payload = {"segments": SEGMENTS, "turns": TURNS, "speaker_names": {"SPEAKER_00": "Anna"}}
    resp = client.post("/v1/transcript", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["utterances"][0]["display_name"] == "Anna"
    assert data["transcript"].startswith("[00:00:00] Anna:")


def test_endpoint_full_service_responses():
    payload = {
        "asr_result": {"text": "Hallo Welt", "language": "de", "segments": SEGMENTS},
        "diarization": {"speakers": TURNS, "number_speakers": 2},
    }
    resp = client.post("/v1/transcript", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["language"] == "de"
    assert data["metadata"]["diarization_provided"] is True


def test_endpoint_text_format():
    payload = {"segments": SEGMENTS, "turns": TURNS, "format": "text"}
    resp = client.post("/v1/transcript", json=payload)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text.splitlines()[0].startswith("[00:00:00] Sprecher 1:")


def test_endpoint_missing_segments_is_400():
    resp = client.post("/v1/transcript", json={"turns": TURNS})
    assert resp.status_code == 400
