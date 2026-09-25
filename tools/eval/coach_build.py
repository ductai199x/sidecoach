"""The coach the evaluation runs, built as the app builds it (`cli._Coaching`).

The eval had its own Coach(...) calls, and they had drifted from the app's: no C in the
system prompt, no rating, verbalized sampling on (the app has it off), the dialogue folded
every 20 plies (the app: never), no claims check (the eval audit, 2026-09-25). A number
measured on that coach is not a number about the coach Tai ships. Here the settings the
app runs with decide all of it -- the system prompt (persona, rating, the shipped C),
verbalized sampling, local verdicts, the summary, the compact record, the verifier, the
routine effort, the claims check, the stuck time -- and every model call the coach can make
goes through the ledger (`providers.Metered`).

What differs from the app, on purpose, and only here:
  - the engine for the moves a player names is whatever the caller gives (the eval's
    CachedEngine, so a replay costs no engine time and repeats exactly);
  - `seed` seeds the coach's wording bags, so a rebuilt corpus says the same templates;
  - `system` replaces the system prompt, for an experimental C only (icr_play.py's
    `--experimental-c`); None is the shipped one.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from providers import Metered  # noqa: E402

from sidecoach import cli  # noqa: E402
from sidecoach.coach.bag import Bags  # noqa: E402
from sidecoach.coach.coach import LESSON_WAIT_S, Coach  # noqa: E402
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings  # noqa: E402
from sidecoach.config import Settings, load_settings  # noqa: E402
from sidecoach.llm.offline import OfflineProvider  # noqa: E402


def build_coach(
    provider: Any = None,
    *,
    ledger: Any = None,
    settings: Settings | None = None,
    engine: Any = None,
    seed: str | int | None = None,
    system: str | None = None,
    summarise_every: int | None = None,
    summariser: Any = None,
    compact_record: bool | None = None,
) -> Coach:
    """The app's coach on `provider` (None or an offline provider: the templates only).

    A paid provider must come with a `ledger` (or already be `Metered`): the coach's own
    calls, the routine effort's, the claims extractor's, a separate verifier's and the
    summariser's are each metered under their own label. `summarise_every`,
    `compact_record` and `summariser` override the app's settings for an experiment; left
    None they are the app's."""
    settings = settings or load_settings()
    inner = getattr(provider, "inner", provider) if provider is not None else OfflineProvider()
    offline = bool(getattr(inner, "offline", False))
    if not offline and ledger is None:
        ledger = getattr(provider, "ledger", None)
        if ledger is None:
            raise ValueError("a paid model needs a ledger: every call is on the record")

    def metered(p: Any, label: str) -> Any:
        if p is None or offline or isinstance(p, Metered):
            return p
        return Metered(p, ledger, label)

    if offline:
        main = inner
    elif isinstance(provider, Metered):
        main = provider
    else:
        main = Metered(inner, ledger, "coach")
    routine = cli._routine_provider(settings, inner)
    routine = main if routine is inner else metered(routine, "coach-routine")
    verifier = None if offline else cli._verifier(settings, main)
    if verifier is not None and verifier is not main:
        verifier = metered(verifier, "verifier")
    # The player's claims are read at the routine effort (none), as the app reads them.
    claims = (
        None
        if offline
        else cli._claims_check(metered(cli._routine_provider(settings, inner), "claims"))
    )
    if summariser is not None and not isinstance(summariser, Metered):
        summariser = metered(summariser, "summariser")
    coach = Coach(
        main,
        policy=InterjectionPolicy(
            PolicySettings(stuck_seconds=settings.coach.stuck_seconds, quiet=settings.coach.quiet)
        ),
        draft_check=cli._draft_check(),
        claims=claims,
        verifier=verifier,  # type: ignore[arg-type]
        engine=engine,
        summarise_every=(
            settings.coach.summarise_every if summarise_every is None else summarise_every
        ),
        summariser=summariser,
        compact_record=settings.coach.compact_record if compact_record is None else compact_record,
        routine_provider=routine,  # type: ignore[arg-type]
        verbalized_sampling=settings.coach.verbalized_sampling,
        local_verdicts=settings.coach.local_verdicts,
        # The app holds a mistake's lesson for the analysis of the position it left.
        lesson_wait_s=LESSON_WAIT_S,
        persona=settings.coach.persona,
        rating=(settings.profile.rating_low, settings.profile.rating_high),
    )
    # Replay and evaluation: the claims check and the summary on the caller's thread, so a
    # run repeats; live they are threaded so the pipeline never waits.
    coach.sync_claims = True
    coach.sync_summaries = True
    if seed is not None:
        coach._bags = Bags(random.Random(str(seed)))  # noqa: SLF001 -- the eval's seed
    if system is not None:
        coach.system = system
    return coach


def coach_config(coach: Coach, settings: Settings | None = None) -> str:
    """One line on what the coach runs with, for a run's log."""
    settings = settings or load_settings()
    c = settings.coach
    return (
        f"coach as the app: persona {c.persona}, rating {settings.profile.rating_low}-"
        f"{settings.profile.rating_high}, verbalized sampling {coach.verbalized_sampling}, "
        f"local verdicts {coach.local_verdicts}, summarise every {coach.summarise_every}, "
        f"verify {c.verify or 'off'}, routine effort {settings.llm.routine_effort}, stuck "
        f"{coach.policy.settings.stuck_seconds:g}s, system prompt ~{len(coach.system) // 4}"
        " tokens"
    )
