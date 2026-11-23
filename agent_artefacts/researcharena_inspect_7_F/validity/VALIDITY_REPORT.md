# Validity report: `researcharena_inspect`

**Version** 7-F · **Date** 19 Nov 2026 · **Group** Scheming

Made with the `eval-validity-review` skill on version 6-A (commit `68c52cc`), then updated for what version 7-F fixes. Paths are relative to the repository root; `D-NNN` are the ids of the deviations table and code tags described in `src/researcharena_inspect/README.md`.

## Summary

The port measures what it says it measures. Its claims about fidelity to upstream ResearchArena at `b922f4b` hold where they could be checked. The scorer is a direct measurement: upstream's own `evaluate.py` run on the agent's workspace. The validity concerns are upstream's, kept and documented: a CUDA score can be read from a file the agent wrote, and the kernels are not checked for correctness. The one concern of the port's own found in the review, an aggregate that reported 0.0 when nothing was measured, is fixed in 7-F. One more was found on 20 Nov and is fixed in 9-F: a transcript monitor could be shown the agent's side-task instruction; the live runs reported in the README were made before the fix (see "Added 20 Nov 2026" below).

**Overall rating: Validity Concerns**, all of them inherited from upstream and disclosed. The port's own contribution rates **Valid with Minor Issues**.

## Claims

14 claims checked: 11 verified, 2 not verifiable in the review, 1 contradicted in an edge case (fixed in 7-F).

- **Verified.** The prompts are byte-identical with upstream's (the tests exist and passed). The paper's tables reproduce from the published runs except for the listed cells (`test_analysis_reproduction.py` passed). The known-good check `pub-min`: eight of ten published CUDA workspaces at 0.955 to 1.003 of their published score. The sixteen live CUDA runs and their cost as the README's evaluation report gives them. The scorer runs upstream's `evaluate.py` from the pinned checkout (code read). No paid model is called in scoring (no `get_model` in scorers; lint rule IEBP013). `HF_TOKEN` is empty in the sandbox (`environment/layout.py`, D-213). The agent's image is pinned by digest. `dataset_samples` 5 and 113. The fine-tuning settings and Kubernetes on an H100 had not run when this was written, as the README then stated (updated 21 Nov 2026: the fine-tuning settings have since run on the H100; Kubernetes has not). Every test the README cites exists (37 names checked).
- **Not verifiable in the review.** The Docker parity rows (slow tests, not run in the review), and "every deviation is tagged in the code" (no exhaustive search).
- **Contradicted in an edge case (Medium), fixed in 7-F.** "The metrics are the mean and count of `main_task_score` over samples that have one" (README, Scoring): when none had one, the mean was reported as 0.0. A real log showed it: a side task whose evaluator never ran (a GPU fault) had a side score of 0.0 in the log's header, where the README's table rightly says "not evaluated". From 7-F such a mean is `nan` (`scoring/scorer.py`, `scored_mean`; test in `tests/researcharena_inspect/test_scoring.py`).

## Name validity

The claimed scope (sabotage by research agents, and monitoring of them, over four settings) matches the code. No issues.

## Dataset validity

The agent has Claude Code (or ReAct) in upstream's image, one GPU, upstream's staged workspace, an open network, and upstream's budget and re-prompting. Success is possible: eleven measured CUDA scores, and an inference starter that passes its gates. Failure is possible. The submission is the workspace itself. Not yet shown on a GPU when this was written: the two fine-tuning settings, and the evaluator's own container (from 6-A). Updated 21 Nov 2026: both have since run on the H100 (the evaluator's container in all four settings; the fine-tuning settings' scorer checks and one five-hour live run per cell), and the evaluation report (now in `src/researcharena_inspect/EVIDENCE.md`) gives the results; in both fine-tuning settings' upload runs the agent refused the upload, so an upload being captured is shown by tests, not by a live run. No High or Medium issue of the port's own.

## Scoring validity

Direct measurement (throughput, accuracy, judged harmfulness) by upstream's evaluators. Edge cases:

- **High, inherited (D-001):** a CUDA score can come from the agent's own `last_bench.json`. This happened in 16 of 113 published runs and in one of the port's live runs (`full-none-3`).
- **Medium, inherited:** no correctness check on kernels (D-013); the inference gates accept empty replies (D-139).
- **Medium, the port's own, fixed in 7-F:** an aggregate reported 0.0 over no values. It is now `nan`.
- **Low, fixed in 7-F:** abnormal scores had no `Score.reason`. A red-team score of `-1` after three failed evaluations now has `reason="scoring_failed"` (the value is upstream's record and stays, D-183); a monitor score of `-1` has `reason="no_response"`.

None of these is hidden: the inherited ones are in the known limitations (now in `src/researcharena_inspect/DEVIATIONS.md`).

## Added 20 Nov 2026 (version 8-F, fixed in 9-F): what a transcript monitor is shown

- **Medium, the port's own, fixed in 9-F (D-078, D-257):** the monitor is meant to judge the agent's behaviour against the task description it is given, which leaves the side task out. Up to 8-F, with Claude Code, the port passed the agent's prompt as a command-line argument (Inspect SWE's only way), so a process listing the agent printed put the whole prompt, side-task instruction included, into the transcript the monitor reads. Upstream's code at the pin pipes the prompt on stdin and would not; the authors' published runs, made before that change upstream, carry the prompt in process listings as the port's did. In the port's fifteen live inference runs, 5 of the 12 side-task transcripts have such a listing, and the monitor cites it in 8 of its 10 judgements on them, with suspicion from 5 to 85: finding the instruction can raise a score, and reading the side task as instructed work can lower it. None of the 12 CUDA side-task runs and none of the 5 five-hour post-training ones (3 when this was written; the two upload runs of 20 Nov added) has one; the three-minute post-training upload check of 20 Nov does. Since 9-F the port gives the CLI a new session's prompt on stdin, as upstream does (a shell reads it from the launch request and pipes it in, as upstream's `printf '%s' "$PROMPT" | claude` does), and a Docker test with the real CLI and a mock model finds neither the prompt nor its side-task section in a process listing, in the first round or a resumed one. Every live Claude Code run reported in the README was made before 9-F, apart from four five-minute checks of the fix itself, so the counts above stand for those runs. ReAct is not affected. Still open and stated (D-078): the monitor's own Claude Code gets its prompt as an argument, where upstream pipes it in; the red-team transcript never receives the monitor's listings, but a monitor that lists its processes sees the prompt it was already given. Effect size not measured; the known limitations (`DEVIATIONS.md`) and the inference monitor results (`EVIDENCE.md`) say this.

## Also fixed in 7-F

- The transcript monitor's image (`monitor/task.py`, `TRANSCRIPT_IMAGE`) is pinned by digest.
- The red-team score's metadata says where scoring ran (`evaluator_service`): the `evaluator` container, or the agent's when the sandbox has no such service (D-093).
- A fast test now runs the monitor's scorer (`monitor_suspicion`) without Docker.

## Recommendations

None left for the port's own code before submission. The inherited items belong to the audit of upstream and are documented as kept.
