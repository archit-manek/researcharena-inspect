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

Updated 22 Nov 2026 (`ca94ce2`, version 10-F). Everything above this section is the 19 Nov review of 6-A and its 7-F fixes, kept as history; its checklist totals (57 met, 18 not met, 4 not yet, 12 not applicable) and its 84% line coverage are that review's and were not recounted for 10-F. The current exceptions to the checklist are listed in the eval README's "Known limitations" ("Inspect Evals checklist: the exceptions of this submission").

- ~~The trajectory analysis (Inspect Scout over the evaluation report's logs): it calls a paid model and waits for approval.~~ Done 21 Nov 2026 (version 10-F; logs made at 1-C to 8-F): `agent_artefacts/trajectory_analysis/researcharena_inspect/`, summarised in the eval README's "Trajectory analysis". The checklist's row (at most 10% invalid samples), after the checks by hand: Claude Code pooled, 4 to 6 of 43 (9.3% to 14.0%), so the threshold is not shown to be met (it holds only if both unsettled failures are valid); capabilities 1 of 5 after the hand check (20%), above the threshold on five samples (CUDA kernels too: 3 to 5 of 16); ReAct 0 to 1 of 4. These invalid failures are upstream's behaviour reproduced by the port (D-037, D-160, D-179). Three inference successes are not confirmed by replay (D-255) and are kept as successes, outside that range.
- ~~A second model in the evaluation report.~~ Done 21 and 22 Nov 2026: GPT-5.5 under ReAct, one half-hour baseline a setting at 10-F (`7f15cba`), each scored in the evaluator's own container (eval README, "What has been verified", and each setting's results). Still missing: a trajectory analysis of the GPT-5.5 runs (the one above covers Sonnet 4.6 only, at 1-C to 8-F), and any GPT-5.5 run with a side task.
- The evaluation report cannot be regenerated from the repository: no `report_config.yaml`, and the logs are held by the author. `src/researcharena_inspect/EVIDENCE.md` maps each table to its logs.
- ~~GPU runs of the two fine-tuning settings and of the evaluator's own container.~~ Done 19 to 21 Nov 2026 on the H100: scorer checks, one live run per cell, ReAct runs and replays of both settings, all scored in the evaluator's own container, which has also run for `cuda_kernel` and `inference_optimization` (eval README, "What has been verified" and "The two post-training settings: what ran"). Still not run: a second run of any cell, and Kubernetes for any setting but `cuda_kernel` (one ten-minute L4 run on 17 Nov, at 1-A).
- Smaller items judged not worth doing before submission: no `stderr` metric beside the means; unknown dict-score leaves are `None`, not `nan`; `small_model` is a parameter, not a model role; some 9 to 17 s tests are not marked slow.
- Reporting the defects found in the authors' data and code to them.
