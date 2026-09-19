# Follow-up design and diagnostic audit

This folder contains a post-hoc analysis of the completed study and a revised design for new experiments. **No v2 model generations, new activation measurements, human ratings or prospective results have been produced.**

- [Saved-output diagnostic audit](V1_DIAGNOSTIC_AUDIT.md), with [machine-readable results and input hashes](V1_DIAGNOSTIC_AUDIT.json).
- [Current design](PROTOCOL.md): test useful steering, benchmark sensitivity, prediction of actual activation collateral, and its relationship to behavior under matched intervention schedules.
- [Amendments](AMENDMENTS.md) explain corrections to the [superseded initial draft](PROTOCOL_DRAFT_2026-09-19_INITIAL.md). Read the current design for active guidance.

The original criterion and all v1 results are retained. The new analysis is exploratory and its intervals are pointwise and conditional on selected features/prompts. The current plan is not an external preregistration or evidence of a new successful method.

Run the audit from the repository root in the documented research environment:

```bash
python -B behavioral_followup_v2/audit_v1.py
```

It reads saved v1 artifacts, reproduces the original primary sentiment contrast, checks feature-level gains, and verifies that v1 files remain unchanged. It writes its JSON and Markdown reports only in this folder. The checked-file count may differ between the original local research folder and a publication checkout containing additional documentation; numerical results must agree. Bootstrap code and input hashes are recorded in the JSON report.
