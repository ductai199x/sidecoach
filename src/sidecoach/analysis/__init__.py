"""Analysis stage: Stockfish multi-PV, expected-score loss, threats, motifs, openings.

Consumes PositionChanged and MovePlayed. Emits Analysis and MoveAssessment. All thresholds
are in expected score, never centipawns. See DESIGN.md section 5.4.
"""
