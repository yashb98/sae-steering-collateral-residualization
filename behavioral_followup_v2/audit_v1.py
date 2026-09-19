"""Read-only, post-hoc diagnostics of v1; writes only into this v2 directory.

This is not a new confirmatory analysis. All intervals are pointwise and
conditional on the selected features and saved prompt population.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / "behavioral_followup"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(setting, bootstrap, prompts):
    folder = V1 / "results" / setting
    names = ["test.jsonl", "test_manifest.json", "selection.json", "analysis.json",
             "development_features.csv", "test_features.csv", "blind_judge_scored.csv"]
    hashes = {name: sha(folder / name) for name in names}
    saved = json.loads((folder / "analysis.json").read_text())
    selection = json.loads((folder / "selection.json").read_text())
    manifest = json.loads((folder / "test_manifest.json").read_text())
    assert hashes["selection.json"] == manifest["selection_sha256"]
    assert hashes["test.jsonl"] == saved["test_sha256"]
    frame = pd.read_json(folder / "test.jsonl", lines=True)
    assert frame.job_id.is_unique and len(frame) == manifest["n_jobs"] == 3648
    reviews = frame[frame.kind == "review"]
    qa = frame[frame.kind == "qa"]
    pivot = reviews.groupby(["feature", "prompt_id"]).positive_probability.mean().unstack()
    assert pivot.shape == (19, 64) and not pivot.isna().any().any()
    assert (reviews.groupby(["feature", "prompt_id"]).size() == 2).all()
    baseline = pivot.loc[-1]
    development = pd.read_csv(folder / "development_features.csv")
    development = development[development.alpha == selection["alpha"]].set_index("feature")
    published_features = pd.read_csv(folder / "test_features.csv").set_index("feature")
    arms = {}
    for arm in ["low", "high", "random"]:
        ids = sorted(frame[frame.arm == arm].feature.unique())
        gains = pivot.loc[ids].subtract(baseline, axis="columns")
        np.testing.assert_allclose(gains.mean(axis=1), published_features.loc[ids, "sentiment_gain"], atol=1e-12)
        gain_interval = bootstrap(gains.to_numpy())
        gain_interval["n_features"] = gain_interval.pop("n_pairs")
        arms[arm] = {
            "test_gain_vs_unsteered": gain_interval,
            "selected_development_mean_gain": float(development.loc[ids, "gain"].mean()),
            "features_with_positive_test_point_estimate": int((gains.mean(axis=1) > 0).sum()),
            "features_with_test_gain_at_least_5pp": int((gains.mean(axis=1) >= .05).sum()),
            "n_features": len(ids),
        }
    # Reconstruct the original primary contrast as a cross-check, not a new test.
    differences = np.array([(pivot.loc[p["low"]] - pivot.loc[p["high"]]).to_numpy()
                            for p in selection["pairs"]])
    reconstructed = bootstrap(differences)
    for key in ["difference", "ci_low", "ci_high"]:
        assert abs(reconstructed[key] - saved["low_minus_high"]["positive_probability"][key]) < 1e-12
    base_qa = qa[qa.feature == -1].set_index("prompt_id")
    comparisons = qa[qa.feature != -1].merge(
        base_qa[["correct", "text"]], left_on="prompt_id", right_index=True,
        suffixes=("", "_baseline"), validate="many_to_one")
    assert len(comparisons) == 18 * 64
    changes = comparisons.correct != comparisons.correct_baseline
    judge = pd.read_csv(folder / "blind_judge_scored.csv")
    checked = judge.merge(reviews[["job_id", "degenerate"]], on="job_id", validate="one_to_one")
    templates = {p["id"]: p["prompt"].replace(p["item"], "{item}")
                 for p in prompts["test"] if p["kind"] == "review"}
    by_template = []
    for template in sorted(set(templates.values())):
        ids = sorted(k for k, v in templates.items() if v == template)
        by_template.append({"template": template, "n_prompts": len(ids),
                            "baseline_positive_probability": float(baseline.loc[ids].mean()),
                            "arm_gain_point_estimates": {
                                arm: float(pivot.loc[sorted(frame[frame.arm == arm].feature.unique()), ids]
                                           .subtract(baseline.loc[ids], axis="columns").to_numpy().mean())
                                for arm in ["low", "high", "random"]}})
    result = {
        "setting": setting, "input_sha256": hashes, "n_outputs": len(frame),
        "arms": arms, "original_primary_contrast_reproduced": True,
        "factual": {"baseline_accuracy": float(base_qa.correct.mean()),
                    "n_feature_prompt_comparisons": len(comparisons),
                    "correctness_changes": int(changes.sum()),
                    "broken_baseline_correct": int((comparisons.correct_baseline.astype(bool) & ~comparisons.correct.astype(bool)).sum()),
                    "changed_full_continuation_text": int((comparisons.text != comparisons.text_baseline).sum())},
        "judge_coverage": {"n_outputs": len(judge),
                           "n_prompt_ids": int(judge.prompt_id.nunique()),
                           "n_templates": len({templates[p] for p in judge.prompt_id}),
                           "total_test_templates": len(set(templates.values())),
                           "nondegenerate_but_judged_incoherent": int((~checked.degenerate & ~checked.coherent).sum()),
                           "caveat": "Automated judge on a fixed homogeneous subset; not human truth."},
        "template_diagnostics": by_template,
    }
    assert hashes == {name: sha(folder / name) for name in names}
    return result


def interval(item):
    return f"{100*item['difference']:+.2f} pp [{100*item['ci_low']:+.2f}, {100*item['ci_high']:+.2f}]"


def main():
    # Import the saved bootstrap routine without invoking its report-writing main.
    spec = importlib.util.spec_from_file_location("v1_analysis", V1 / "analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = {str(p.relative_to(V1)): sha(p) for p in V1.rglob("*")
              if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    prompts = json.loads((V1 / "prompts.json").read_text())
    reports = [audit(s, module.bootstrap_pairs, prompts) for s in ["gpt2_small", "gemma_2_2b"]]
    after = {str(p.relative_to(V1)): sha(p) for p in V1.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    assert before == after, "The completed v1 artifacts must remain unchanged."
    result = {"date": "2026-09-19", "status": "post-hoc exploratory diagnostic; no new generations",
              "bootstrap": "5000 feature/prompt resamples; two review seeds averaged within prompt; pointwise 95% intervals",
              "source_sha256": {"audit_v1.py": sha(Path(__file__)), "v1_analyze.py": sha(V1 / "analyze.py"),
                                "v1_prompts.json": sha(V1 / "prompts.json")},
              "v1_unchanged_files_checked": len(before), "settings": reports}
    (ROOT / "V1_DIAGNOSTIC_AUDIT.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Diagnostic audit of the completed behavioral study", "", "19 September 2026. Post-hoc exploratory analysis of saved outputs; no new model generations. These analyses were not preregistered and do not replace the original criterion or results.", "",
             "The key distinction is intended steering versus unsteered performance, separately from low/high noninferiority. Intervals below resample selected features and prompt identities (5,000 draws), retaining review seeds together. They are pointwise, conditional intervals, with no multiplicity correction or guarantee of generalization to new feature populations.", "",
             "| Model | Arm | Development gain | Test gain vs unsteered, 95% CI | Features with test gain >=5 pp |", "|---|---|---:|---:|---:|"]
    for r in reports:
        for arm, a in r["arms"].items():
            lines.append(f"| {r['setting']} | {arm} | {100*a['selected_development_mean_gain']:+.2f} pp | {interval(a['test_gain_vs_unsteered'])} | {a['features_with_test_gain_at_least_5pp']}/{a['n_features']} |")
    lines += ["", "Development gains are selected estimates; their change on test mixes selection optimism, prompt differences and sampling variation. This audit does not identify their separate causes.", ""]
    for r in reports:
        f, j = r["factual"], r["judge_coverage"]
        lines += [f"## {r['setting']}", "", f"Factual correctness changed in {f['correctness_changes']}/{f['n_feature_prompt_comparisons']} feature/prompt comparisons. Full continuation text changed in {f['changed_full_continuation_text']} comparisons; this includes text after the scored answer and is not a harm measure.", "",
                  f"The secondary judge covered {j['n_prompt_ids']} prompt IDs and {j['n_templates']}/{j['total_test_templates']} review templates. {j['nondegenerate_but_judged_incoherent']}/{j['n_outputs']} judged outputs passed the simple nondegeneration filter but were rated incoherent. The judge is another model, not human ground truth.", ""]
    lines += ["## Consequences for the next study", "", "- Establish intended benefit against an unsteered baseline on fresh validation data, in addition to low/high noninferiority.", "- Validate that an independently chosen control intervention produces measurable factual errors at usable output quality. A perfect unsteered baseline does not prevent measuring degradation.", "- Cover distinct prompt templates and add blinded human review before making semantic-quality claims.", "- Measure downstream activation collateral and behavioral errors under matched prompts and intervention schedules. The saved outputs cannot establish that link without new activation measurements.", "- Plan sample size using pilot variance across independent feature pairs and prompt families. Repeated generations are not independent feature replications.", "", f"Verification: both original primary sentiment contrasts and feature-level sentiment gains reproduce within 1e-12; all {len(before)} checked v1 files remained byte-identical. Full input hashes and descriptive template results are in [V1_DIAGNOSTIC_AUDIT.json](V1_DIAGNOSTIC_AUDIT.json).", ""]
    (ROOT / "V1_DIAGNOSTIC_AUDIT.md").write_text("\n".join(lines))
    print(json.dumps({"checked_v1_files": len(before), "settings": [{"setting": r["setting"], "arms": r["arms"], "factual": r["factual"], "judge_coverage": r["judge_coverage"]} for r in reports]}, indent=2))


if __name__ == "__main__":
    main()
