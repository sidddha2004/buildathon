export type ReviewDecision = "approve" | "review" | "recapture"

export type EvidenceSignal = {
  label: string
  value: string
  strength: "match" | "mismatch" | "uncertain"
  source: string
}

export type ReturnCase = {
  id: string
  orderId: string
  customer: string
  product: string
  category: string
  amount: number
  receivedAt: string
  outbound: { sku: string; serial: string | null; weightGrams: number }
  returned: { sku: string; serial: string | null; weightGrams: number }
  features: RiskFeatures
  label: "genuine" | "substitution" | "unknown"
  evidence: EvidenceSignal[]
}

export type RiskFeatures = {
  visionSimilarity: number
  vlmMismatch: number
  serialMismatch: number
  weightDelta: number
  imageQuality: number
}

export type RiskResult = {
  probability: number
  score: number
  decision: ReviewDecision
  reasons: string[]
}

export type EvaluationSummary = {
  sampleSize: number
  positiveRate: number
  precision: number
  recall: number
  f1: number
  falsePositives: number
  falseNegatives: number
  falsePositiveCost: number
  missedLoss: number
}

const sigmoid = (value: number) => 1 / (1 + Math.exp(-value))

export function scoreReturn(features: RiskFeatures, threshold = 0.68): RiskResult {
  const logit =
    -3.15 +
    (1 - features.visionSimilarity) * 4.1 +
    features.vlmMismatch * 2.75 +
    features.serialMismatch * 2.3 +
    Math.min(features.weightDelta / 0.35, 1) * 1.65

  const probability = sigmoid(logit)
  const reasons: string[] = []

  if (features.visionSimilarity < 0.68) reasons.push("Visual identity is materially different from dispatch evidence")
  if (features.vlmMismatch > 0.62) reasons.push("Vision-language verifier found product-level discrepancies")
  if (features.serialMismatch > 0.7) reasons.push("Serial or model identifier does not match the order record")
  if (features.weightDelta > 0.18) reasons.push("Parcel weight differs beyond the configured tolerance")
  if (features.imageQuality < 0.46) reasons.push("Image quality is below the evidence threshold; capture clearer product photos")
  if (reasons.length === 0) reasons.push("All available return evidence is consistent with dispatch")

  const decision: ReviewDecision =
    features.imageQuality < 0.46 ? "recapture" : probability >= threshold ? "review" : "approve"

  return { probability, score: Math.round(probability * 100), decision, reasons }
}

export const returnCases: ReturnCase[] = [
  {
    id: "RET-1042", orderId: "ORD-78421", customer: "Aarav K.", product: "Auralite Pro Earbuds", category: "Electronics", amount: 8499, receivedAt: "11:42",
    outbound: { sku: "AL-PRO-BLK", serial: "AL9-82K-104", weightGrams: 286 },
    returned: { sku: "AL-PRO-BLK", serial: "AL7-19M-552", weightGrams: 198 },
    features: { visionSimilarity: 0.58, vlmMismatch: 0.88, serialMismatch: 1, weightDelta: 0.308, imageQuality: 0.94 },
    label: "substitution",
    evidence: [
      { label: "Visual identity", value: "58% match", strength: "mismatch", source: "DINOv2 pair" },
      { label: "Serial OCR", value: "Different", strength: "mismatch", source: "OCR binding" },
      { label: "Parcel weight", value: "−88 g", strength: "mismatch", source: "Warehouse scan" },
      { label: "Image quality", value: "Sufficient", strength: "match", source: "Quality gate" },
    ],
  },
  {
    id: "RET-1041", orderId: "ORD-78409", customer: "Mira S.", product: "Northstar Trail Shoe", category: "Footwear", amount: 4799, receivedAt: "11:17",
    outbound: { sku: "NS-TRAIL-39", serial: null, weightGrams: 914 },
    returned: { sku: "NS-TRAIL-39", serial: null, weightGrams: 922 },
    features: { visionSimilarity: 0.94, vlmMismatch: 0.12, serialMismatch: 0, weightDelta: 0.009, imageQuality: 0.9 },
    label: "genuine",
    evidence: [
      { label: "Visual identity", value: "94% match", strength: "match", source: "DINOv2 pair" },
      { label: "SKU & variant", value: "Exact match", strength: "match", source: "Order record" },
      { label: "Parcel weight", value: "+8 g", strength: "match", source: "Warehouse scan" },
      { label: "Image quality", value: "Sufficient", strength: "match", source: "Quality gate" },
    ],
  },
  {
    id: "RET-1039", orderId: "ORD-78366", customer: "Kabir N.", product: "Eon Smartwatch S2", category: "Electronics", amount: 12999, receivedAt: "10:56",
    outbound: { sku: "EON-S2-GRY", serial: "ES2-003-991", weightGrams: 342 },
    returned: { sku: "EON-S2-GRY", serial: null, weightGrams: 338 },
    features: { visionSimilarity: 0.7, vlmMismatch: 0.48, serialMismatch: 0.35, weightDelta: 0.012, imageQuality: 0.31 },
    label: "unknown",
    evidence: [
      { label: "Visual identity", value: "Inconclusive", strength: "uncertain", source: "DINOv2 pair" },
      { label: "Serial OCR", value: "Unreadable", strength: "uncertain", source: "OCR binding" },
      { label: "Parcel weight", value: "−4 g", strength: "match", source: "Warehouse scan" },
      { label: "Image quality", value: "Insufficient", strength: "uncertain", source: "Quality gate" },
    ],
  },
  {
    id: "RET-1036", orderId: "ORD-78321", customer: "Ishita R.", product: "Morrow Linen Shirt", category: "Apparel", amount: 2899, receivedAt: "10:31",
    outbound: { sku: "ML-SEA-M", serial: null, weightGrams: 384 },
    returned: { sku: "ML-SEA-L", serial: null, weightGrams: 410 },
    features: { visionSimilarity: 0.66, vlmMismatch: 0.72, serialMismatch: 0, weightDelta: 0.068, imageQuality: 0.86 },
    label: "substitution",
    evidence: [
      { label: "Visual identity", value: "66% match", strength: "mismatch", source: "DINOv2 pair" },
      { label: "SKU & variant", value: "Size differs", strength: "mismatch", source: "Order record" },
      { label: "Parcel weight", value: "+26 g", strength: "uncertain", source: "Warehouse scan" },
      { label: "Image quality", value: "Sufficient", strength: "match", source: "Quality gate" },
    ],
  },
]

function seededRandom(seed: number) {
  let state = seed >>> 0
  return () => {
    state = (1664525 * state + 1013904223) >>> 0
    return state / 4294967296
  }
}

export function evaluateSyntheticBaseline(threshold = 0.68, seed = 5050): EvaluationSummary {
  const random = seededRandom(seed)
  const sampleSize = 480
  let tp = 0, fp = 0, fn = 0, positives = 0

  for (let index = 0; index < sampleSize; index += 1) {
    const isSwap = random() < 0.18
    if (isSwap) positives += 1
    const hardNegative = !isSwap && random() < 0.14
    const features: RiskFeatures = isSwap
      ? {
          visionSimilarity: 0.43 + random() * 0.42,
          vlmMismatch: 0.42 + random() * 0.52,
          serialMismatch: random() < 0.44 ? 1 : random() * 0.28,
          weightDelta: random() * 0.34,
          imageQuality: 0.52 + random() * 0.48,
        }
      : {
          visionSimilarity: hardNegative ? 0.58 + random() * 0.25 : 0.79 + random() * 0.2,
          vlmMismatch: hardNegative ? 0.28 + random() * 0.42 : random() * 0.3,
          serialMismatch: random() < 0.025 ? 0.82 : random() * 0.12,
          weightDelta: hardNegative ? random() * 0.19 : random() * 0.09,
          imageQuality: 0.52 + random() * 0.48,
        }

    const predictedSwap = scoreReturn(features, threshold).decision === "review"
    if (predictedSwap && isSwap) tp += 1
    if (predictedSwap && !isSwap) fp += 1
    if (!predictedSwap && isSwap) fn += 1
  }

  const precision = tp / Math.max(tp + fp, 1)
  const recall = tp / Math.max(tp + fn, 1)
  return {
    sampleSize,
    positiveRate: positives / sampleSize,
    precision,
    recall,
    f1: (2 * precision * recall) / Math.max(precision + recall, Number.EPSILON),
    falsePositives: fp,
    falseNegatives: fn,
    falsePositiveCost: fp * 80,
    missedLoss: fn * 6200,
  }
}
