# Results and claim boundaries

## Confirmed main result

On fresh plain prompts, the cat-trained/neutral/mix50 counts are 396/10/12 out of 720 each. Baseline excess is 53.61 pp (family interval [34.86,71.94]); defended excess is 0.28 pp. Absolute reduction is 53.33 pp ([34.58,71.67]); relative reduction is 99.48% ([98.32,100.00]). Per-seed baseline excess: 32.50,72.92,55.42 pp; mix50 excess: 0.42,0.42,0.00 pp. Both frozen success criteria pass.

GSM8K cat accuracy is 519/600=86.50%, mix50 is 533/600=88.83%, neutral is 501/600=83.50%, and base is 167/200=83.50%. These are three evaluations on the SAME 200 problems, not 600 distinct problems. Paired loss is -2.33 pp, interval [-5.17,+0.33]. The interval includes zero: no established improvement claim.

## Robustness, nulls, and failures

Seed-1 plain baseline gap is 32.50 pp. It falls to 1.25 pp with 2,000 rows, 2.08 pp with one epoch, and 1.67 pp after canonical formatting. These are single-seed paired comparisons. Training exposure differs with size/duration; canonicalization changes tokenization. They support sensitivity, not a localized causal mechanism.

The six fresh plain families show baseline gaps from 18.33 to 85.83 pp. The residual mix50 gap is concentrated in dictionary examples (1.67 pp); the other five are zero at the observed sample size. Zero counts do not bound unseen or rare differences.

JSON baseline/defended excess: 31.39/-0.28 pp; relative suppression 100.88% is retained without clipping. TAG: 11.11/0.14 pp; suppression 98.75%. TAG validity is only 187/720 cat, 257/720 neutral, and 209/720 mix50; strict output compliance and preference expression are confounded. Plain validity is 717/720,720/720,720/720; JSON validity 720/720,717/720,716/720. JSON/TAG seed-resampling ratio intervals are unavailable because 5/8 draws have zero denominators; absolute intervals remain in the archived result.

The frequency filter failed dev eligibility and was not confirmed; its paired gap (12.75 pp) exceeded the unfiltered dev baseline (9.50 pp). Neutral continuation was eligible on dev but not selected. Do not promote development comparisons into held-out findings or describe mix50 as the universally best defense.

## What this study does not establish

- Generalization across architectures, sizes, traits, languages, or harmful behaviors.
- A universal or adaptive-attack-resistant safety defense.
- A mechanistic explanation or independent validation of token-entanglement claims.
- Capability preservation beyond one 200-question math damage screen.
- Robust output-format generalization given poor TAG compliance.
- An unchanged preregistration: disclose the approved post-hoc dev casing amendment.
- Independence of prompts, completions, training seeds, or a random representative family sample.
- A matched equal-token compute comparison across every intervention.

mix50 requires access to paired neutral completions. Neutrality is defined by this generation setup, not guaranteed for an arbitrary external corpus. One model/trait, three main seeds, six hand-designed families, and one filtering mask limit external validity. More experiments are optional future work, not required to manufacture a milestone-to-final progress story.
