"""Merge von ASR-Segmenten und Diarization-Turns zu einem Sprecher-Protokoll.

Reine Funktionen ohne I/O, damit sie sich direkt unit-testen lassen
(siehe processor-service/tests/test_transcript.py).

Eingabe:
  * ASR-Segmente:    [{"start": 72.1, "end": 76.4, "text": "..."}, ...]
  * Diarization:     [{"start": 72.0, "end": 78.2, "speaker": "SPEAKER_00"}, ...]

Ausgabe:
  * Segmente mit Sprecher-Zuordnung (an Sprecherwechseln gesplittet)
  * Gruppierte Aeinanderfolge gleicher Sprecher ("Aeusserungen")
  * Formatierter Protokolltext:  [hh:mm:ss] Sprecher 1: Text
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Segment = Dict[str, object]  # {"start": float, "end": float, "text": str}
Turn = Dict[str, object]  # {"start": float, "end": float, "speaker": str}


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign_speaker(
    start: float, end: float, turns: Sequence[Turn]
) -> Optional[str]:
    """Sprecher mit der groessten zeitlichen Ueberlappung zum Intervall.

    Faellt das Intervall komplett in eine Pause (keine Ueberlappung), wird der
    zeitlich naechste Turn uebernommen - sonst entstehen "Unbekannt"-Bruchstuecke
    zwischen zwei Sprecherbeitraegen.
    """
    best: Optional[str] = None
    best_overlap = 0.0
    for turn in turns:
        ov = _overlap(start, end, float(turn["start"]), float(turn["end"]))
        if ov > best_overlap:
            best_overlap = ov
            best = str(turn["speaker"])
    if best_overlap > 0:
        return best
    if not turns:
        return None

    center = (start + end) / 2.0

    def _distance(turn: Turn) -> float:
        t0, t1 = float(turn["start"]), float(turn["end"])
        if t0 <= center <= t1:
            return 0.0
        return min(abs(center - t0), abs(center - t1))

    nearest = min(turns, key=_distance)
    return str(nearest["speaker"])


def split_interval(
    start: float, end: float, turns: Sequence[Turn]
) -> List[Tuple[float, float, Optional[str]]]:
    """Zerlegt [start, end] an Sprecherwechseln (Turn-Grenzen innenhalb)."""
    cuts = sorted(
        {
            c
            for turn in turns
            for c in (float(turn["start"]), float(turn["end"]))
            if start + 1e-6 < c < end - 1e-6
        }
    )
    points = [start] + cuts + [end]
    spans: List[Tuple[float, float, Optional[str]]] = []
    for a, b in zip(points, points[1:]):
        if b - a <= 1e-6:
            continue
        spans.append((a, b, assign_speaker(a, b, turns)))
    if not spans:
        spans = [(start, end, assign_speaker(start, end, turns))]
    return spans


def _split_text(text: str, spans: Sequence[Tuple[float, float, object]]) -> List[str]:
    """Verteilt den Text anteilig nach Dauer der Teil-Intervalle.

    Es wird an Wortgrenzen geteilt (nie mitten im Wort), damit das Protokoll
    lesbar bleibt.
    """
    if not spans:
        return []
    if not text:
        return ["" for _ in spans]
    if len(spans) == 1:
        return [text]

    words = text.split()
    if not words:
        return ["" for _ in spans]

    durations = [max(1e-9, float(b) - float(a)) for a, b, _ in spans]
    total = sum(durations)
    n = len(words)
    parts: List[str] = []
    prev_idx = 0
    acc = 0.0
    for i, dur in enumerate(durations):
        acc += dur
        if i == len(durations) - 1:
            idx = n
        else:
            idx = int(round(n * acc / total))
        idx = max(prev_idx, min(n, idx))
        parts.append(" ".join(words[prev_idx:idx]))
        prev_idx = idx
    return parts


def merge_segments(
    segments: Iterable[Segment], turns: Sequence[Turn]
) -> List[Segment]:
    """Weist jedem ASR-Segment einen Sprecher zu und splittet an Wechseln."""
    merged: List[Segment] = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        text = str(seg.get("text") or "").strip()
        spans = split_interval(start, end, turns)
        texts = _split_text(text, spans)
        for (a, b, speaker), part in zip(spans, texts):
            if not part:
                continue
            merged.append(
                {
                    "start": round(a, 3),
                    "end": round(b, 3),
                    "speaker": speaker,
                    "text": part,
                }
            )
    return merged


def _join_texts(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith("-"):
        return left + right
    return f"{left} {right}"


def group_utterances(segments: Sequence[Segment]) -> List[Segment]:
    """Gruppiert aufeinanderfolgende Segmente desselben Sprechers."""
    utterances: List[Segment] = []
    for seg in segments:
        speaker = seg.get("speaker")
        if (
            utterances
            and utterances[-1].get("speaker") == speaker
            and speaker is not None
        ):
            utterances[-1]["end"] = max(
                float(utterances[-1]["end"]), float(seg["end"])
            )
            utterances[-1]["text"] = _join_texts(
                str(utterances[-1]["text"]), str(seg["text"])
            )
        else:
            utterances.append(dict(seg))
    return utterances


def display_names(
    utterances: Sequence[Segment], speaker_names: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Sprecher-Reihenfolge nach erstem Auftreten; Umbenennung per mapping."""
    speaker_names = speaker_names or {}
    order: List[str] = []
    for utt in utterances:
        speaker = utt.get("speaker")
        if speaker and speaker not in order:
            order.append(speaker)
    names: Dict[str, str] = {}
    for i, speaker in enumerate(order):
        names[speaker] = speaker_names.get(speaker) or f"Sprecher {i + 1}"
    return names


def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_transcript(
    utterances: Sequence[Segment], names: Dict[str, str]
) -> str:
    """Protokoll im Format [hh:mm:ss] Sprecher: Text."""
    lines = []
    for utt in utterances:
        speaker = utt.get("speaker")
        if speaker is None:
            display = "Unbekannt"
        else:
            display = names.get(str(speaker), str(speaker))
        lines.append(f"[{_timestamp(float(utt['start']))}] {display}: {utt['text']}")
    return "\n".join(lines)


def speaker_summary(
    utterances: Sequence[Segment], names: Dict[str, str]
) -> List[Dict[str, object]]:
    summary: List[Dict[str, object]] = []
    index: Dict[str, Dict[str, object]] = {}
    for utt in utterances:
        speaker = utt.get("speaker")
        if not speaker:
            continue
        entry = index.get(str(speaker))
        if entry is None:
            entry = {
                "speaker": str(speaker),
                "display_name": names.get(str(speaker), str(speaker)),
                "first_appearance": round(float(utt["start"]), 3),
                "utterances": 0,
                "speech_sec": 0.0,
            }
            index[str(speaker)] = entry
            summary.append(entry)
        entry["utterances"] = int(entry["utterances"]) + 1
        entry["speech_sec"] = round(
            float(entry["speech_sec"]) + (float(utt["end"]) - float(utt["start"])), 3
        )
    return summary


def build_transcript(
    segments: Sequence[Segment],
    turns: Sequence[Turn],
    speaker_names: Optional[Dict[str, str]] = None,
    language: Optional[str] = None,
) -> Dict[str, object]:
    """Kompletter Merge: Segmente -> Aeusserungen -> Protokolltext."""
    merged = merge_segments(segments, turns)
    utterances = group_utterances(merged)
    names = display_names(utterances, speaker_names)
    for utt in utterances:
        speaker = utt.get("speaker")
        utt["display_name"] = (
            names.get(str(speaker), str(speaker)) if speaker else "Unbekannt"
        )

    return {
        "segments": merged,
        "utterances": utterances,
        "speakers": speaker_summary(utterances, names),
        "transcript": format_transcript(utterances, names),
        "language": language,
        "counts": {
            "asr_segments": len(list(segments)),
            "merged_segments": len(merged),
            "utterances": len(utterances),
            "turns": len(list(turns)),
        },
    }
