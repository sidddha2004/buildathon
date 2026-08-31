export type HealthResponse = {
  status: string
  policy: string
  models: { dinov2: string; qwen3_vl: string; fusion: string }
  auditor: { mode: "llm_api" | "deterministic_fallback"; model: string | null; authority: string }
}

export type VerificationResponse = {
  risk: {
    probability: number
    score: number
    decision: "approve" | "review" | "recapture"
    reasons: string[]
  }
  features: {
    vision_similarity: number
    vlm_mismatch: number
    serial_mismatch: number
    weight_delta: number
    image_quality: number
  }
  vlm_assessment: {
    evidence_sufficient: boolean
    same_product_likelihood: number
    mismatch_confidence: number
    observations: Array<{
      attribute: string
      dispatch_value: string
      return_value: string
      severity: "minor" | "material"
      evidence_ids: string[]
    }>
    missing_evidence: string[]
  }
  quality: {
    dispatch: { score: number; brightness: number; contrast: number; sharpness: number }
    return: { score: number; brightness: number; contrast: number; sharpness: number }
  }
  evidence_sources: string[]
  policy_note: string
  auditor_assessment: {
    recommendation_support: "supported" | "needs_more_evidence" | "contradictory"
    evidence_consistent: boolean
    contradictions: string[]
    missing_evidence: string[]
    reviewer_summary: string
    checked_evidence_ids: string[]
    source: "llm_api" | "deterministic_fallback"
    api_status: "used" | "not_configured" | "failed"
    model: string | null
    latency_ms: number
    authority: "advisory_only"
  }
}

export type VerificationForm = {
  dispatchImage: File
  returnImage: File
  dispatchSku: string
  returnSku: string
  dispatchSerial?: string
  returnSerial?: string
  dispatchWeight?: string
  returnWeight?: string
}

export const defaultApiUrl =
  process.env.NEXT_PUBLIC_SWAPSHIELD_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000"

function responseError(payload: unknown, fallback: string): string {
  if (!payload || typeof payload !== "object" || !("detail" in payload)) return fallback
  const detail = (payload as { detail: unknown }).detail
  if (typeof detail === "string") return detail
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) return String(item.msg)
        return String(item)
      })
      .join("; ")
  }
  return fallback
}

export async function fetchHealth(apiUrl: string): Promise<HealthResponse> {
  const response = await fetch(`${apiUrl.replace(/\/$/, "")}/health`, { cache: "no-store" })
  if (!response.ok) throw new Error(`Health check failed with HTTP ${response.status}`)
  return response.json() as Promise<HealthResponse>
}

export async function verifyReturn(
  apiUrl: string,
  values: VerificationForm,
): Promise<VerificationResponse> {
  const form = new FormData()
  form.append("dispatch_image", values.dispatchImage)
  form.append("return_image", values.returnImage)
  form.append("dispatch_sku", values.dispatchSku.trim())
  form.append("return_sku", values.returnSku.trim())
  if (values.dispatchSerial?.trim()) form.append("dispatch_serial", values.dispatchSerial.trim())
  if (values.returnSerial?.trim()) form.append("return_serial", values.returnSerial.trim())
  if (values.dispatchWeight?.trim()) form.append("dispatch_weight_grams", values.dispatchWeight.trim())
  if (values.returnWeight?.trim()) form.append("return_weight_grams", values.returnWeight.trim())

  let response: Response
  try {
    response = await fetch(`${apiUrl.replace(/\/$/, "")}/v1/verify`, { method: "POST", body: form })
  } catch {
    throw new Error("The local GPU API is unreachable. Start FastAPI and open the dashboard locally.")
  }
  const payload = (await response.json().catch(() => null)) as unknown
  if (!response.ok) throw new Error(responseError(payload, `Verification failed with HTTP ${response.status}`))
  return payload as VerificationResponse
}
