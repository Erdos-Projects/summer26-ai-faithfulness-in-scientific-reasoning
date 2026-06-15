# Rubric Scoring (DeLeAn demand annotation)

Routine, rubric-driven feature extraction over SciVer figure-based claim-verification
items, following the ADeLe/DeLeAn method (Zhou et al. 2026, Nature). Each item is scored
on demand rubrics — one scale at a time, in isolation — on the credit-free Claude Code
subagent path, with per-call token/cost provenance.

## Design

See **[`../docs/RUBRIC_SCORING_SKILL_DESIGN.md`](../docs/RUBRIC_SCORING_SKILL_DESIGN.md)** —
the validated design spec (scope, 6 first-pass dims, architecture, prompt, data model,
multi-pass/resume, deviations, cost/time, portability). The production `SKILL.md` and
portable pipeline will be built under `.claude/skills/` per the implementation plan.

## Layout

- `rubrics/` — DeLeAn rubric texts extracted verbatim from the supplementary
  (currently the 6 piloted: AS, MCr, QLq, QLl, VO, AT; the library will grow to all 18).
- `pilot/` — **historical pilot record**, not production code:
  - `annotations.db` — SQLite with 5 pilot passes (`pilot_0001`–`pilot_0005`, 18 cells each;
    `skill_version = pilot-0.1`). Demonstrates the multi-run schema and stability analysis.
  - `pilot_setup.py`, `scrape_pilot.py` — pilot scaffolding. **Contain Erdos-absolute paths**
    and are kept only as a record; the production pipeline will be path-portable.
  - `*_manifest.json`, `pilot_prompts/` — the exact tagged prompts used per pilot pass.

## Pilot findings (3 SciVer chart items × 6 rubrics × 5 passes)

- Mechanism works end-to-end; 100% score parse rate.
- Per agent: ~47–50k tokens processed, ~20 s, ~$0.055–0.06 API-equivalent ($0 actual on Max).
- Stability across 5 passes: 82% mean modal agreement, 89% within ±1, 68% mean pairwise exact.
- Reasoning verified OFF (0 thinking tokens); temperature 0 and a hard output cap are not
  controllable on the subagent path — documented and measured rather than assumed.
- Full-run projection (charts only, 817 × 6 = 4,902 agents/pass): ~$0 actual, a multi-session
  rate-limited campaign.
