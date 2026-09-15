"""Coach stage: disclosure state machine, interjection policy, claim verification, LLM turns.

Consumes Analysis, MoveAssessment, UserUtterance, clocks. Emits CoachMessage and
ProfileEvidence. Hiding is by construction: see `disclosure.py`. DESIGN.md section 5.5.
"""
