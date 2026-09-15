"""Vision stage: board detection, square classification, orientation, OCR.

Consumes Frame. Emits BoardLocated, BoardLost, SquareObservation, ClockObservation,
ScreenStateChanged. Board detection is classical CV over the checkerboard's regularities;
a learned detector is only a fallback. See DESIGN.md section 5.2.
"""
