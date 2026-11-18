# Review summary: `researcharena_inspect`

19 Nov 2026, version 6-A (`68c52cc`) reviewed; fixes shipped in version 7-F. `D-NNN` ids refer to the deviations table in `src/researcharena_inspect/README.md`.

## What was checked

Inspect Evals' EVALUATION_CHECKLIST (91 rows), BEST_PRACTICES and TASK_VERSIONING, with the `eval-validity-review` and `ensure-test-coverage` skills. Run: `inspect-evals-lint --all`, `make check`, `make test` (520 passed, 171 skipped as slow or Docker), and line coverage of the fast suite (84%). Read only: the slow and Docker tests, and the GPU results the README reports.

## What was found

Lint, checks and tests passed, and nothing found contradicts what the README claims. The checklist came out at 57 met, 18 not met, 4 not yet (needs runs) and 12 not applicable. The findings that mattered:

1. A mean over no values reported 0.0, seen in a real log as a side score of 0.0 for a side task that was never evaluated.
2. The task version's letter went backwards (3-D, 4-C, 5-A, 6-A), and the README had no changelog.
3. The evaluation report has one model with no reason given, and no trajectory analysis.
4. This file and the validity report were missing.
5. Abnormal scores carried no `Score.reason`.
6. The transcript monitor's image was a mutable tag.
7. No `NOTICE`, and the files adapted from upstream did not name its licence.
8. The fast suite never ran the monitor's scorer.

The upstream flaws the port keeps on purpose (D-001, D-013, D-139, D-183) were not counted as findings.

## What was fixed in 7-F

1. `nan`, not 0.0, over no values; a real 0.0 still reports 0.0.
2. Version 7-F; a changelog in the README, which says the letters of 4-C to 6-A were out of order.
3. The README's report says why it has one model (time and compute) and that the trajectory analysis is not done yet.
4. `agent_artefacts/researcharena_inspect_7_F/validity/VALIDITY_REPORT.md` and this file.
5. `reason="scoring_failed"` and `reason="no_response"` on the two `-1` scores; values unchanged.
6. The transcript monitor's and the CPU test image pinned by digest.
7. A root `NOTICE` and an "Adapted from" line in each adapted file.
8. A fast test of `monitor_suspicion` with the fake sandbox.

Also: the red-team score's metadata records which container scored (`evaluator_service`), and stale wording left by the 6-A container change was corrected.

## What is left

- The trajectory analysis (Inspect Scout over the evaluation report's logs): it calls a paid model and waits for approval.
- A second model in the evaluation report.
- GPU runs of the two fine-tuning settings and of the evaluator's own container.
- Smaller items judged not worth doing before submission: no `stderr` metric beside the means; unknown dict-score leaves are `None`, not `nan`; `small_model` is a parameter, not a model role; some 9 to 17 s tests are not marked slow; `report_config.yaml` is absent.
- Reporting the defects found in the authors' data and code to them.
