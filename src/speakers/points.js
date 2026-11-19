// Operating points of the speaker assignment: the thresholds measured for a model file. They belong to that file only
// (its sha256 in the catalog); never reuse them with another model.

/** Stricter point of the fine-tuned Czech model (measured 2026-11-07). */
export const STRICT = Object.freeze({
  id: 'voiceprint-cs/strict/2026-11-07',
  named: 0.38,
  lengthSlope: 0.05,
  margin: 0,
  session: 0.28,
  minMs: 1000,
  drop: 0.4,
  splitWindowMs: 1500,
  splitHopMs: 500,
  enrollWindowMs: 3000,
  enrollHopMs: 1000,
});

/** Point of the original (base) model, only to reproduce its measurement; it is not in the catalog. */
export const BASE_DEPLOYABLE = Object.freeze({
  id: 'eres2net-base/deployable',
  named: 0.57,
  lengthSlope: 0.16,
  margin: 0,
  session: 0.34,
  minMs: 1000,
  drop: 0.2,
  splitWindowMs: 1500,
  splitHopMs: 500,
  enrollWindowMs: 3000,
  enrollHopMs: 1000,
});
