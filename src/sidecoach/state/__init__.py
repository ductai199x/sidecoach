"""State stage: reconciliation into a legal python-chess game.

Consumes vision events. Emits GameStarted, PositionChanged, MovePlayed, Takeback, Resync,
GameEnded. Detects moves over the legal-move set rather than reconstructing FEN per frame.
See DESIGN.md section 5.3.
"""
