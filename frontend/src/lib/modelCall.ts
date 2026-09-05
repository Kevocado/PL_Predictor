// A "model call" is a plain, informational probability decisively on one
// side or the other -- distinct from a value bet (which requires a live
// market the model beats). Threshold is a starting judgment call (see
// docs/superpowers/specs/2026-09-04-bet-recommendation-trustworthiness-design.md),
// open to tuning after real usage.
export function isModelCall(prob: number): boolean {
  return prob >= 0.6 || prob <= 0.4;
}
