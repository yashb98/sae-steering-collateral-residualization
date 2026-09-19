# Behavioral follow-up v2: testing the link from activation spread to behavioral harm

Status, 19 September 2026: design revision and saved-output audit completed; no v2 model generations, activation measurements or held-out results exist. This is a locally dated exploratory design, not an external preregistration. The executable pilot and its input manifest are the next deliverables. Freeze the final evaluation protocol only after a separate calibration pilot.

## Research question and motivation

Under matched contexts and intervention schedules, does predicted SAE activation collateral identify interventions that preserve unrelated behavior at comparable, demonstrably useful intended steering?

The completed September 18 study remains unchanged. It found no factual correctness differences across arms. The new [saved-output audit](V1_DIAGNOSTIC_AUDIT.md) additionally compares selected arms with the unsteered baseline: the pointwise 95% sentiment-gain intervals for low-score features include zero in both models. Noninferiority to high-score features alone therefore does not establish useful steering. Development-to-test changes do not identify whether selection optimism, prompt shift or sampling variation is responsible.

This new design is informed by v1 outcomes. That is legitimate exploratory learning, but v1 outcomes are not fresh confirmation. Do not claim that v1 results did not influence the redesign. The [initial draft](PROTOCOL_DRAFT_2026-09-19_INITIAL.md) is retained for provenance; its ceiling explanation and numerical power claims are superseded by [AMENDMENTS.md](AMENDMENTS.md).

## Highest-value distinction

Measure three separate quantities:

1. The existing pre-intervention collateral forecast, frozen before new outcomes.
2. Actual downstream SAE collateral, measured under the tested context, strength and schedule.
3. Intended behavioral gain and factual/coherence loss, measured on generated outputs.

If the forecast misses actual activation collateral, the problem includes prediction transfer. If actual collateral is predicted well but is uninformative about validated behavioral harm, the activation metric may be an inadequate behavioral proxy in that setting. If the benchmark cannot register a usable control intervention's damage, the behavioral test is inconclusive. These explanations can coexist; correlations alone do not establish a causal mediation mechanism.

A comparison of original final-prompt-token-only steering with final-prompt-token plus generated-position steering tests one concrete distribution change. The original predictor was trained on final-token activation-count labels, while v1 evaluated repeated generation-time steering.

## Stage 1: bounded calibration pilot, before another full study

Start with GPT-2-small to check instrumentation cheaply; then check feasibility on Gemma-2-2B. A successful small-model pilot is not evidence of large-model transfer.

- Retain the frozen v1 predictor and original SAE definitions. Do not train it on any behavioral outcome.
- Sample a small panel across predicted-collateral ranks, independently of v1 test outcomes. Exclude v1 held-out selected features from the prospective confirmation pool; retain explicit feature IDs and their prior use. If this leaves insufficient independent candidates, expand discovery with a documented rule before generation.
- Separate pilot prompts, policy-selection prompts, calibration prompts and final-test prompts by both content and template family where practical. Record exact and normalized overlap checks. Reused discovery data must be disclosed; new wording alone is not an independent domain replication.
- Evaluate a short, fixed coefficient grid and both intervention schedules. Freeze the grid, feature IDs, output limits, seeds and a compute cap in the pilot manifest before inference. Stop increasing strength when output-quality requirements fail; do not force a high dose merely to manufacture a preservation difference.
- Test contextual retrieval with controlled distractors first. Parametric short-answer QA is an optional separately reported benchmark only if the unsteered base model can answer enough questions. Neither benchmark is assumed valid before measurement.
- Include unsteered outputs, a separate predetermined set of norm-matched random directions, and random eligible SAE features. Random vectors and random sentiment-associated features answer different control questions. Select neither control set using final-test outcomes.
- On separate calibration data, require evidence that an independently chosen control intervention changes factual correctness while maintaining usable output quality. Report broken originally correct answers and repaired originally wrong answers. Baseline accuracy of 100% can still reveal damage; an arbitrary upper baseline-accuracy ceiling is not a sensitivity test.
- Demonstrate positive intended steering against unsteered outputs on calibration data, with a declared practical target and quality margin. A proposed starting target is a 5-percentage-point sentiment gain; its suitability and the semantic-quality rubric must be resolved using the pilot and then frozen. This is not a retrospectively achieved criterion.
- Judge calibration must include mixed sentiment, negation, incoherent positive-word text, off-topic text and factual contradictions. Obvious adjective examples alone do not validate generated-text scores.

If either useful steering or benchmark sensitivity cannot be established, report the failed gate and investigate that cause. Do not spend a full test budget on a selector-benefit claim with an unvalidated outcome.

## Stage 2: measure the forecast-to-activation link

Use the same feature, prompt, coefficient and schedule for predictor evaluation and the measured downstream effect. Save the original raw collateral count and normalized metric separately, together with logit-effect magnitude, direction norm, firing frequency and natural activation.

For repeated steering, first use teacher-forced common unsteered continuation tokens: clean and perturbed passes then share token histories. Predefine the measured positions and averaging rule. Free-running clean and steered generations have different prefixes after divergence; activation differences there mix steering effects with changed text and must be reported as a separate diagnostic. Teacher-forced measurements do not themselves establish generated behavioral harm.

Evaluate the frozen forecast against observed collateral on a broader, independently sampled eligible feature panel, as well as the sentiment-selected subset. Restriction to a small selected subset can reduce score range. Compare frequency-only, activation-only, crowding-only and the combined forecast on identical held-out feature/context splits. Fit any new predictor or propagation baseline only on designated training data, with comparable access to labels and tuning budget; retain the original frozen predictor's results separately.

## Stage 3: practical benefit at matched intended success

Use development data to choose the feature/strength policy and match low/high groups on intended gain and quality. Match steering utility, not just decoder norm or coefficient. Lock the policy before fresh calibration and final-test outcomes. If held-out success differs materially, report the utility/side-effect tradeoff; do not rematch on test outcomes or call weaker steering safer.

The proposed final decision requires all three conditions, with the endpoint, margin, interval method and multiplicity family frozen before final evaluation:

1. Intended benefit versus unsteered outputs, not only noninferiority to another steered arm.
2. Comparable intended success for the compared policies, with quality requirements satisfied.
3. A factual-preservation advantage on a sensitivity-validated benchmark.

The original v1 two-part criterion is unchanged in its original report. The above is a prospective strengthening for a different experiment, not a relabeling of v1.

Use a diverse blinded assessment covering all planned prompt families and arms. Prepare a randomized annotation sheet and rubric for two independent human reviewers; actual human ratings and adjudication remain pending. An automated second judge is supplementary and must not be described as human validation.

## Stage 4: sample size, replication and reporting

Use pilot estimates of feature-pair variation, prompt/template variation, baseline/steered discordance and their correlations in a design simulation. Check false-positive rate as well as power for the exact proposed decision rule, including sensitivity to larger-than-observed variances. Then fix the number of independent feature pairs, prompt families and replications for a practically meaningful effect. There is currently no justified numerical power or minimum-detectable-effect claim for v2.

Repeated seeds or many prompts for six pairs do not produce many independent feature replications. Plan inference around the policy or feature population actually claimed. Very few feature clusters require particular caution with bootstrap intervals. Keep an independent confirmation block of features if claiming generalization to new features.

Replicate a frozen result on a second model and a distinct prompt domain. Multiple feature sampling seeds, independent corpus draws and independently trained SAE dictionaries are different robustness questions; do not substitute one for another. Dictionary width variation is already in Evan's paper and is not by itself a new contribution.

Publish the protocol history, configurations, manifests, raw outputs, scores, exclusions and verification. Keep null/adverse results and failed gates. Public timestamping before the final test would improve auditability; no external preregistration has yet been made. The original manuscript contribution can proceed independently while this extension is investigated.

## Positioning and immediate deliverables

The potentially distinctive contribution is a controlled account of when an activation-side predictor transfers to behavioral preservation, including identified limits under context or schedule changes. Novelty is a hypothesis to check, not a claim established by this plan. General side-effect forecasting and dose control already have prior work.

Immediate completed deliverables: reproducible saved-output audit, corrected design and updated email. Next: implement and smoke-test deployment-matched activation logging; build and hash fresh pilot prompts and control IDs; run calibration; use its variance estimates to finish the frozen evaluation and sample-size plan. The full evaluation design remains incomplete until calibration and sample-size planning are finished.

Parallel manuscript work: align directly with Evan's feature rows, downstream panels and nuisance-variable definitions when supplied. That is the most direct route to strengthening the current revision; the new behavioral study should not delay sending the completed evidence.

## Primary-source basis

- [Duan et al., Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects](https://arxiv.org/html/2606.08365v1): original final-token intervention and activation-side target.
- [Ong et al., Forecasting Side Effects of Activation Steering](https://arxiv.org/html/2608.11227v1): existing behavioral forecasting and dose-controlled analyses; broad forecasting is not an available first claim.
- [Are Sparse Autoencoder Benchmarks Reliable?](https://arxiv.org/html/2605.18229v1): distinguishes measurement noise, validity and discriminative power; more seeds do not resolve every validity gap.

These sources motivate the design, not an inference that it will produce a positive result. The current source check is not an exhaustive novelty review.
