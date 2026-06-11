# On-Policy Distillation Examples

The canonical OPD documentation lives in
[`docs/advanced/on-policy-distillation.md`](../../docs/advanced/on-policy-distillation.md).
Keep the algorithm description, arguments, teacher-mode comparison, and
Rethinking OPD top-k recipe there so we do not maintain two copies.

This directory contains runnable examples:

- `run-qwen3-8B-opd.sh`: SGLang teacher server OPD. This script enables
  Rethinking OPD with `--opd-log-prob-top-k 16`, `--opd-top-k-strategy only_stu`,
  and `--opd-reward-weight-mode student_p`.
- `run-qwen3-8B-opd-multi-teacher.sh`: Multi-teacher OPD with per-sample routing.
  Math prompts are scored by a Qwen3-32B teacher and code prompts by a
  Qwen3-Coder-30B-A3B teacher, selected via `--opd-teacher-urls` and a per-row
  `{"metadata": {"opd_teacher": ...}}` tag in the dataset.
- `run-qwen3-8B-opd-multi-teacher-smoke.sh`: Cheap routing smoke test using two
  open datasets (DAPO -> teacher A, GSM8K -> teacher B, untagged rows ->
  `default`). Both teachers serve the student's own weights, so
  `opd_reverse_kl ~ 0` doubles as a logprob-alignment oracle.
- `run-qwen3-8B-opd-ensemble.sh`: Teacher-ensemble OPD. Every sample is scored
  by a weighted group of two teachers in parallel and the targets are combined
  as a probability-space mixture (`--opd-teacher-urls` with comma-separated
  URLs and `@weight` suffixes), using the exact tail-bucket top-k KL
  (`--opd-topk-tail-bucket`).
- `run-qwen3-8B-opd-megatron.sh`: Megatron-loaded teacher OPD.

Use `--opd-log-prob-top-k 0` to run the original sampled-token OPD path.
