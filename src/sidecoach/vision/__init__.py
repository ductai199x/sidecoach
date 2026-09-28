"""Vision stage: what the frame shows of the board.

Board detection and tracking (board_detect, tracker), the pieces' recognition (pieces,
embedding, theme_cache), the clocks, the coordinate labels and the result badges. The state
stage's pipeline turns what they read into events. Board detection is classical CV over the
checkerboard's regularities. See DESIGN.md section 5.2.
"""
