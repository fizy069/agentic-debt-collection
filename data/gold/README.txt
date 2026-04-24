Gold Set for Meta-Evaluation (Darwin Godel Machine)

This directory contains hand-labeled expected scores for the eval judges.
The meta-evaluator uses these as ground truth to measure judge accuracy
and detect blind spots.

labels/<scenario_id>.json
  Expected scores for quality and handoff judges. Compliance ground truth
  comes from the deterministic checks + persona-implied rules, so it does
  not need gold labels.

Filenames must match the scenario_id format from scenarios.py:
  {persona_type}-{seed}-{index:02d}
  e.g. cooperative-42-00, distressed-42-00, combative-42-00

Key test case:
  distressed-42-00 — borrower mentions job loss without using the literal
  word "hardship". Tests whether the compliance judge correctly identifies
  rule 5 as triggered. This is the seeded blind-spot for the DGM demo.
