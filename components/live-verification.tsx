"use client"

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react"
import {
  AlertTriangle, BrainCircuit, Check, CheckCircle2, CircleDot, FileImage, ImageOff,
  LoaderCircle, LockKeyhole, Printer, RefreshCw, ScanLine, Server, ShieldCheck, Upload,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import {
  defaultApiUrl, fetchHealth, verifyReturn, type HealthResponse, type VerificationResponse,
} from "@/lib/live-api"

type ConnectionState = "checking" | "online" | "offline"
type HumanDecision = "approve" | "recapture" | "review"

function usePreview(file: File | null) {
  const url = useMemo(() => file ? URL.createObjectURL(file) : null, [file])
  useEffect(() => {
    return () => {
      if (url) URL.revokeObjectURL(url)
    }
  }, [url])
  return url
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`
}

function decisionMeta(decision: VerificationResponse["risk"]["decision"]) {
  if (decision === "approve") return { label: "Approve recommendation", tone: "safe", icon: CheckCircle2 }
  if (decision === "recapture") return { label: "Request recapture", tone: "uncertain", icon: ImageOff }
  return { label: "Route to human review", tone: "risk", icon: AlertTriangle }
}

export function LiveVerification() {
  const [apiUrl, setApiUrl] = useState(defaultApiUrl)
  const [connection, setConnection] = useState<ConnectionState>("checking")
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [dispatchImage, setDispatchImage] = useState<File | null>(null)
  const [returnImage, setReturnImage] = useState<File | null>(null)
  const [dispatchSku, setDispatchSku] = useState("")
  const [returnSku, setReturnSku] = useState("")
  const [dispatchSerial, setDispatchSerial] = useState("")
  const [returnSerial, setReturnSerial] = useState("")
  const [dispatchWeight, setDispatchWeight] = useState("")
  const [returnWeight, setReturnWeight] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<VerificationResponse | null>(null)
  const [humanDecision, setHumanDecision] = useState<HumanDecision | null>(null)
  const dispatchPreview = usePreview(dispatchImage)
  const returnPreview = usePreview(returnImage)

  const checkConnection = useCallback(async () => {
    setConnection("checking")
    try {
      const next = await fetchHealth(apiUrl)
      setHealth(next)
      setConnection("online")
    } catch {
      setHealth(null)
      setConnection("offline")
    }
  }, [apiUrl])

  useEffect(() => {
    let active = true
    void fetchHealth(apiUrl).then(
      (next) => {
        if (!active) return
        setHealth(next)
        setConnection("online")
      },
      () => {
        if (!active) return
        setHealth(null)
        setConnection("offline")
      },
    )
    return () => {
      active = false
    }
  }, [apiUrl])

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    setError(null)
    setResult(null)
    setHumanDecision(null)
    if (!dispatchImage || !returnImage) {
      setError("Upload both dispatch and return evidence images.")
      return
    }
    if (!dispatchSku.trim() || !returnSku.trim()) {
      setError("Enter both dispatch and return SKUs.")
      return
    }
    setLoading(true)
    try {
      const next = await verifyReturn(apiUrl, {
        dispatchImage,
        returnImage,
        dispatchSku,
        returnSku,
        dispatchSerial,
        returnSerial,
        dispatchWeight,
        returnWeight,
      })
      setResult(next)
      setConnection("online")
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Verification failed unexpectedly.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="space-y-5">
      <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
        <div>
          <p className="eyebrow">Live local inference / RTX 5050</p>
          <h3 className="mt-2 text-xl font-semibold tracking-tight">Compare real dispatch and return evidence</h3>
          <p className="mt-1 max-w-3xl text-sm leading-6 text-muted-foreground">The calibrated model scores the pair first. A separate LLM API audits whether the structured evidence supports that recommendation. A human owns the final action.</p>
        </div>
        <ConnectionBadge state={connection} health={health} />
      </div>

      <form onSubmit={submit} className="space-y-5">
        <div className="panel p-5 lg:p-6">
          <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
            <div><p className="text-sm font-medium">Local API connection</p><p className="mt-1 text-xs text-muted-foreground">Use the FastAPI address printed by Uvicorn.</p></div>
            <Button type="button" variant="outline" size="sm" onClick={() => void checkConnection()} disabled={connection === "checking"}><RefreshCw className={connection === "checking" ? "animate-spin" : ""} /> Recheck</Button>
          </div>
          <div className="mt-4 grid gap-2 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
            <div><Label htmlFor="api-url">API URL</Label><Input id="api-url" className="mt-2 font-mono" value={apiUrl} onChange={(event) => setApiUrl(event.target.value)} /></div>
            <p className="pb-2 text-xs text-muted-foreground">Hosted showcase may show offline; run both services locally for GPU inference.</p>
          </div>
        </div>

        <div className="grid gap-5 lg:grid-cols-2">
          <UploadCard id="dispatch-image" title="Dispatch evidence" subtitle="Reference captured before fulfilment" file={dispatchImage} preview={dispatchPreview} onChange={setDispatchImage} />
          <UploadCard id="return-image" title="Return evidence" subtitle="Item received back at the warehouse" file={returnImage} preview={returnPreview} onChange={setReturnImage} />
        </div>

        <div className="panel p-5 lg:p-6">
          <div className="flex items-center justify-between"><div><p className="text-sm font-medium">Objective order evidence</p><p className="mt-1 text-xs text-muted-foreground">SKU is required. Serial and weight improve the case when available.</p></div><LockKeyhole className="size-4 text-muted-foreground" /></div>
          <div className="mt-5 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <Field id="dispatch-sku" label="Dispatch SKU *" value={dispatchSku} onChange={setDispatchSku} placeholder="MOUSE-001" />
            <Field id="return-sku" label="Return SKU *" value={returnSku} onChange={setReturnSku} placeholder="MOUSE-001" />
            <Field id="dispatch-serial" label="Dispatch serial" value={dispatchSerial} onChange={setDispatchSerial} placeholder="ABC-123" />
            <Field id="return-serial" label="Return serial" value={returnSerial} onChange={setReturnSerial} placeholder="ABC-123" />
            <Field id="dispatch-weight" label="Dispatch weight (g)" value={dispatchWeight} onChange={setDispatchWeight} placeholder="1000" type="number" />
            <Field id="return-weight" label="Return weight (g)" value={returnWeight} onChange={setReturnWeight} placeholder="995" type="number" />
          </div>
          {error && <div className="mt-5 flex gap-3 rounded-xl border border-red-400/25 bg-red-400/5 p-4 text-sm text-red-200"><AlertTriangle className="mt-0.5 size-4 shrink-0" /><span>{error}</span></div>}
          <div className="mt-6 flex flex-col justify-between gap-3 border-t border-border pt-5 sm:flex-row sm:items-center">
            <p className="text-xs leading-5 text-muted-foreground">First execution may take longer while GPU weights warm up. Keep this page open until the audit is complete.</p>
            <Button type="submit" size="lg" disabled={loading || connection === "offline"}>{loading ? <><LoaderCircle className="animate-spin" /> Running verifier and evidence audit…</> : <><ScanLine /> Verify return</>}</Button>
          </div>
        </div>
      </form>

      {loading && <LoadingPanel auditorMode={health?.auditor.mode} />}
      {result && <LiveResult result={result} dispatchPreview={dispatchPreview} returnPreview={returnPreview} humanDecision={humanDecision} onDecision={setHumanDecision} />}
    </section>
  )
}

function ConnectionBadge({ state, health }: { state: ConnectionState; health: HealthResponse | null }) {
  if (state === "checking") return <Badge variant="outline"><LoaderCircle className="animate-spin" /> Checking local API</Badge>
  if (state === "offline") return <Badge variant="outline" className="border-red-400/30 text-red-300"><Server /> Local API offline</Badge>
  return <Badge variant="outline" className="border-primary/30 text-primary"><span className="size-1.5 rounded-full bg-primary" /> GPU API online · auditor {health?.auditor.mode === "llm_api" ? "API" : "fallback"}</Badge>
}

function UploadCard({ id, title, subtitle, file, preview, onChange }: { id: string; title: string; subtitle: string; file: File | null; preview: string | null; onChange: (file: File | null) => void }) {
  return (
    <div className="panel overflow-hidden">
      <div className="relative flex h-56 items-center justify-center overflow-hidden border-b border-border bg-muted/20" style={preview ? { backgroundImage: `linear-gradient(rgba(4,16,12,.12),rgba(4,16,12,.28)),url(${preview})`, backgroundPosition: "center", backgroundSize: "contain", backgroundRepeat: "no-repeat" } : undefined}>
        {!preview && <div className="text-center text-muted-foreground"><FileImage className="mx-auto size-8" /><p className="mt-3 text-xs">JPEG, PNG or WebP · maximum 12 MB</p></div>}
        {preview && <Badge className="absolute right-3 bottom-3 bg-background/80 text-foreground">Preview</Badge>}
      </div>
      <div className="p-5"><p className="text-sm font-medium">{title}</p><p className="mt-1 text-xs text-muted-foreground">{subtitle}</p><Label htmlFor={id} className="mt-4 flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-3 text-xs transition hover:border-primary/40 hover:text-primary"><Upload className="size-4" />{file ? "Replace image" : "Choose image"}</Label><Input id={id} type="file" accept="image/jpeg,image/png,image/webp" className="sr-only" onChange={(event) => onChange(event.target.files?.[0] ?? null)} /><p className="mt-2 truncate font-mono text-[10px] text-muted-foreground">{file?.name ?? "No file selected"}</p></div>
    </div>
  )
}

function Field({ id, label, value, onChange, placeholder, type = "text" }: { id: string; label: string; value: string; onChange: (value: string) => void; placeholder: string; type?: "text" | "number" }) {
  return <div><Label htmlFor={id}>{label}</Label><Input id={id} className="mt-2" type={type} min={type === "number" ? 0 : undefined} value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} /></div>
}

function LoadingPanel({ auditorMode }: { auditorMode?: HealthResponse["auditor"]["mode"] }) {
  return <div className="panel p-6"><div className="flex items-center gap-3"><LoaderCircle className="size-5 animate-spin text-primary" /><div><p className="text-sm font-medium">Building the evidence chain</p><p className="mt-1 text-xs text-muted-foreground">DINOv2 comparison → Qwen3-VL observations → calibrated fusion → {auditorMode === "llm_api" ? "external LLM audit" : "safe consistency fallback"}</p></div></div><Progress value={72} className="mt-5" /><p className="mt-3 text-xs text-muted-foreground">The progress indicator is staged; model inference time depends on image complexity and GPU warm-up.</p></div>
}

function LiveResult({ result, dispatchPreview, returnPreview, humanDecision, onDecision }: { result: VerificationResponse; dispatchPreview: string | null; returnPreview: string | null; humanDecision: HumanDecision | null; onDecision: (decision: HumanDecision) => void }) {
  const meta = decisionMeta(result.risk.decision)
  const Icon = meta.icon
  const audit = result.auditor_assessment
  return (
    <section className="print-report space-y-5">
      <div className="panel p-5 lg:p-6">
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start"><div><p className="eyebrow">Completed evidence chain</p><h3 className="mt-2 text-xl font-semibold tracking-tight">Reviewer recommendation</h3><p className="mt-1 text-sm text-muted-foreground">The score is produced before the independent evidence audit.</p></div><div className={`risk-banner ${meta.tone}`}><Icon className="size-4" /><span>{meta.label}</span><strong>{result.risk.score}</strong></div></div>
        <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_280px]">
          <ResultImage label="Dispatch evidence" preview={dispatchPreview} />
          <ResultImage label="Return evidence" preview={returnPreview} />
          <div className="rounded-2xl border border-border bg-muted/20 p-4"><p className="text-xs uppercase tracking-wider text-muted-foreground">Calibrated probability</p><p className="mt-3 text-4xl font-semibold tabular-nums">{percent(result.risk.probability)}</p><Progress value={result.risk.score} className={`mt-4 progress-${meta.tone}`} /><p className="mt-4 text-xs leading-5 text-muted-foreground">{result.policy_note}</p></div>
        </div>
        <div className="mt-5 grid gap-3 border-t border-border pt-5 sm:grid-cols-2 lg:grid-cols-5">
          <ResultStat label="Vision similarity" value={percent(result.features.vision_similarity)} />
          <ResultStat label="VLM mismatch" value={percent(result.features.vlm_mismatch)} />
          <ResultStat label="Identifier mismatch" value={percent(result.features.serial_mismatch)} />
          <ResultStat label="Weight deviation" value={percent(result.features.weight_delta)} />
          <ResultStat label="Image quality" value={percent(result.features.image_quality)} />
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.1fr)_minmax(340px,0.9fr)]">
        <div className="panel overflow-hidden"><div className="border-b border-border px-5 py-4"><p className="text-sm font-medium">Qwen3-VL observable differences</p><p className="mt-1 text-xs text-muted-foreground">Evidence extraction only; no decision authority</p></div>{result.vlm_assessment.observations.length ? <Table><TableHeader><TableRow><TableHead>Attribute</TableHead><TableHead>Dispatch</TableHead><TableHead>Return</TableHead><TableHead>Severity</TableHead></TableRow></TableHeader><TableBody>{result.vlm_assessment.observations.map((item, index) => <TableRow key={`${item.attribute}-${index}`}><TableCell className="font-medium">{item.attribute.replaceAll("_", " ")}</TableCell><TableCell className="text-muted-foreground">{item.dispatch_value}</TableCell><TableCell>{item.return_value}</TableCell><TableCell><Badge variant="outline" className={item.severity === "material" ? "border-red-400/30 text-red-300" : ""}>{item.severity}</Badge></TableCell></TableRow>)}</TableBody></Table> : <div className="p-5 text-sm text-muted-foreground">No directly supported visual differences were returned.</div>}</div>

        <div className="panel p-5"><div className="flex items-center justify-between"><div className="flex items-center gap-3"><div className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary"><BrainCircuit className="size-4" /></div><div><p className="text-sm font-medium">Independent evidence auditor</p><p className="text-xs text-muted-foreground">Second opinion · advisory only</p></div></div><Badge variant="outline" className={audit.source === "llm_api" ? "border-primary/30 text-primary" : "border-amber-400/30 text-amber-300"}>{audit.source === "llm_api" ? "LLM API" : "Safe fallback"}</Badge></div><div className="mt-5 rounded-xl border border-border bg-muted/20 p-4"><div className="flex items-center justify-between gap-3"><p className="text-xs uppercase tracking-wider text-muted-foreground">Recommendation support</p><strong className="text-xs uppercase tracking-wider text-primary">{audit.recommendation_support.replaceAll("_", " ")}</strong></div><p className="mt-3 text-sm leading-6">{audit.reviewer_summary}</p></div>{audit.contradictions.length > 0 && <AuditList title="Contradictions" items={audit.contradictions} warning />}{audit.missing_evidence.length > 0 && <AuditList title="Missing evidence" items={audit.missing_evidence} warning />}<div className="mt-4 flex flex-wrap gap-2">{audit.checked_evidence_ids.map((id) => <Badge key={id} variant="outline" className="font-mono text-[10px]">{id}</Badge>)}</div>{audit.api_status !== "used" && <p className="mt-4 text-xs leading-5 text-amber-300">{audit.api_status === "failed" ? "The configured LLM API failed or violated the schema; deterministic fallback was used." : "No auditor API is configured; deterministic consistency checks were used."}</p>}</div>
      </div>

      <div className="panel p-5"><div className="flex flex-col justify-between gap-4 md:flex-row md:items-center"><div><div className="flex items-center gap-2"><ShieldCheck className="size-4 text-primary" /><p className="text-sm font-medium">Human decision</p></div><p className="mt-1 text-xs text-muted-foreground">Neither the fusion model nor the auditor can execute an adverse outcome.</p></div><div className="flex flex-wrap gap-2"><Button type="button" variant={humanDecision === "approve" ? "default" : "outline"} onClick={() => onDecision("approve")}>Approve</Button><Button type="button" variant={humanDecision === "recapture" ? "default" : "outline"} onClick={() => onDecision("recapture")}>Recapture</Button><Button type="button" variant={humanDecision === "review" ? "default" : "outline"} onClick={() => onDecision("review")}>Send to review</Button><Button type="button" variant="outline" onClick={() => window.print()}><Printer /> Print report</Button></div></div>{humanDecision && <p className="mt-4 flex items-center gap-2 border-t border-border pt-4 text-xs text-primary"><Check className="size-3" /> Human decision recorded for this browser session: {humanDecision}.</p>}</div>
    </section>
  )
}

function ResultImage({ label, preview }: { label: string; preview: string | null }) {
  return <div className="overflow-hidden rounded-2xl border border-border bg-muted/20"><div className="h-44 bg-contain bg-center bg-no-repeat" style={preview ? { backgroundImage: `url(${preview})` } : undefined} /><p className="border-t border-border p-3 text-xs font-medium">{label}</p></div>
}

function ResultStat({ label, value }: { label: string; value: string }) {
  return <div><p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p><p className="mt-1 text-sm font-medium tabular-nums">{value}</p></div>
}

function AuditList({ title, items, warning = false }: { title: string; items: string[]; warning?: boolean }) {
  return <div className="mt-4"><p className={`text-xs font-medium ${warning ? "text-amber-300" : ""}`}>{title}</p><ul className="mt-2 space-y-2">{items.map((item) => <li key={item} className="flex gap-2 text-xs leading-5 text-muted-foreground"><CircleDot className="mt-1 size-3 shrink-0 text-amber-300" />{item}</li>)}</ul></div>
}
