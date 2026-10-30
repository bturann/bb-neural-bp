# Project instructions

Read `docs/implementation_spec.md` and `NOTES.md` before changing the research
implementation. The handoff specification is preserved unchanged; documented API
findings and justified implementation deviations live in `NOTES.md`.

- Keep the research question and experiment matrix intact. Obtain human agreement
  before changing the question, dropping an experiment, adding a direct dependency
  outside the handoff list, or launching a run exceeding four GPU-hours.
- Never fabricate or manually edit experimental results. Save resolved configs,
  seeds, source provenance, and failures. Distinguish development from final data.
- Leave `reference/ref_impl.py` and `reference/test_ref.py` unchanged.
- Run `.venv/bin/python -m pytest -q` before each milestone commit. Include the
  slow acceptance tests at milestone completion. Record the milestone's gate
  outcome in `NOTES.md`; do not claim a milestone passed without its evidence.
- Use explicit, config-seeded NumPy or PyTorch generators for random draws.
- Keep Python 3.11 and the pinned environment. Use a writable cache directory in
  restricted workspaces; do not modify system Python.
