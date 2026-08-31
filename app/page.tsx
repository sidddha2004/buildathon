"use client"

import { useMemo, useState } from "react"
import {
  Activity, AlertTriangle, Braces, Check, CheckCircle2, CircleDot, Cpu, FileCheck2,
  Fingerprint, Gauge, ImageOff, LockKeyhole, ScanLine, ShieldCheck, Sparkles, Workflow,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { LiveVerification } from "@/components/live-verification"
import realReport from "@/evaluation/results/real-report.json"
import { evaluateSyntheticBaseline, returnCases, scoreReturn, type ReviewDecision } from "@/lib/swapshield"

const money = new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 })

function decisionMeta(decision: ReviewDecision) {
  if (decision === "approve") return { label: "Approve refund", tone: "safe", icon: CheckCircle2 }
  if (decision === "recapture") return { label: "Request recapture", tone: "uncertain", icon: ImageOff }
  return { label: "Human review", tone: "risk", icon: AlertTriangle }
}

export default function Home() {
  const [selectedId, setSelectedId] = useState(returnCases[0].id)
  const [demoThreshold, setDemoThreshold] = useState(68)
  const [recordedDecision, setRecordedDecision] = useState<ReviewDecision | null>(null)
  const selected = returnCases.find((item) => item.id === selectedId) ?? returnCases[0]
  const result = scoreReturn(selected.features, demoThreshold / 100)
  const syntheticEvaluation = useMemo(() => evaluateSyntheticBaseline(demoThreshold / 100), [demoThreshold])
  const lockedTest = realReport.test
  const recommended = decisionMeta(result.decision)
  const RecommendedIcon = recommended.icon

  function selectCase(id: string) {
    setSelectedId(id)
    setRecordedDecision(null)
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/80 bg-background/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1500px] items-center justify-between px-5 py-4 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex size-10 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-[0_0_30px_rgba(68,224,167,0.18)]"><ShieldCheck className="size-5" /></div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-semibold tracking-tight">SwapShield AI</h1>
                <Badge variant="outline" className="border-primary/30 text-primary">FINAL · v1.0.0</Badge>
              </div>
              <p className="text-xs text-muted-foreground">Return authenticity command center</p>
            </div>
          </div>
          <div className="hidden items-center gap-3 sm:flex">
            <div className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5 text-xs text-muted-foreground">
              <span className="size-1.5 rounded-full bg-primary shadow-[0_0_10px_var(--primary)]" />Defense-only policy active
            </div>
            <Button variant="outline" size="sm" disabled><Activity /> Session audit</Button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1500px] px-5 py-5 lg:px-8 lg:py-7">
        <Tabs defaultValue="queue" className="gap-6">
          <div className="flex flex-col justify-between gap-4 border-b border-border pb-4 md:flex-row md:items-center">
            <div>
              <p className="eyebrow">Risk operations / Returns</p>
              <h2 className="mt-1 text-2xl font-semibold tracking-tight">Verify the item before the refund</h2>
            </div>
            <TabsList variant="line" className="w-full justify-start overflow-x-auto md:w-auto">
              <TabsTrigger value="queue"><ScanLine /> Review queue</TabsTrigger>
              <TabsTrigger value="live"><Sparkles /> Live verify</TabsTrigger>
              <TabsTrigger value="evaluation"><Gauge /> Evaluation</TabsTrigger>
              <TabsTrigger value="pipeline"><Workflow /> Model pipeline</TabsTrigger>
            </TabsList>
          </div>

          <TabsContent value="queue">
            <section className="grid gap-5 xl:grid-cols-[360px_minmax(0,1fr)]">
              <aside className="panel overflow-hidden">
                <div className="flex items-center justify-between border-b border-border px-4 py-4">
                  <div><p className="text-sm font-medium">Incoming returns</p><p className="mt-0.5 text-xs text-muted-foreground">Prioritised by expected loss</p></div>
                  <Badge variant="secondary">4 seeded cases</Badge>
                </div>
                <div className="divide-y divide-border">
                  {returnCases.map((item) => {
                    const itemResult = scoreReturn(item.features, demoThreshold / 100)
                    const meta = decisionMeta(itemResult.decision)
                    const active = selected.id === item.id
                    return (
                      <button key={item.id} type="button" onClick={() => selectCase(item.id)} className={`group w-full px-4 py-4 text-left transition ${active ? "bg-accent" : "hover:bg-accent/50"}`}>
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2"><span className="font-mono text-xs text-muted-foreground">{item.id}</span><span className={`risk-dot ${meta.tone}`} /></div>
                            <p className="mt-2 truncate text-sm font-medium">{item.product}</p>
                            <p className="mt-1 text-xs text-muted-foreground">{item.customer} · {item.receivedAt}</p>
                          </div>
                          <div className="text-right"><p className="text-lg font-semibold tabular-nums">{itemResult.score}</p><p className="text-[10px] uppercase tracking-wider text-muted-foreground">risk</p></div>
                        </div>
                        <div className="mt-3 flex items-center justify-between text-xs"><span className={`decision-text ${meta.tone}`}>{meta.label}</span><span className="text-muted-foreground">{money.format(item.amount)}</span></div>
                      </button>
                    )
                  })}
                </div>
              </aside>

              <div className="space-y-5">
                <section className="panel p-5 lg:p-6">
                  <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                      <div className="flex flex-wrap items-center gap-2"><Badge variant="outline" className="font-mono">{selected.id}</Badge><span className="text-xs text-muted-foreground">{selected.orderId}</span><span className="text-xs text-muted-foreground">•</span><span className="text-xs text-muted-foreground">{selected.category}</span></div>
                      <h3 className="mt-3 text-xl font-semibold tracking-tight">{selected.product}</h3>
                      <p className="mt-1 text-sm text-muted-foreground">Return value {money.format(selected.amount)} · received today at {selected.receivedAt}</p>
                    </div>
                    <div className={`risk-banner ${recommended.tone}`}><RecommendedIcon className="size-4" /><span>{recommended.label}</span><strong>{result.score}</strong></div>
                  </div>

                  <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_300px]">
                    <EvidenceCard title="Dispatch fingerprint" subtitle="Signed at fulfilment" sku={selected.outbound.sku} serial={selected.outbound.serial} weight={selected.outbound.weightGrams} variant="dispatch" />
                    <EvidenceCard title="Returned item" subtitle="Warehouse capture" sku={selected.returned.sku} serial={selected.returned.serial} weight={selected.returned.weightGrams} variant="return" />
                    <div className="rounded-2xl border border-border bg-muted/25 p-4">
                      <div className="flex items-center justify-between"><p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">POC risk score</p><Fingerprint className="size-4 text-primary" /></div>
                      <p className="mt-3 text-4xl font-semibold tabular-nums">{result.score}<span className="text-lg text-muted-foreground">/100</span></p>
                      <Progress value={result.score} className={`mt-4 progress-${recommended.tone}`} />
                      <p className="mt-3 text-xs leading-5 text-muted-foreground">Demo threshold is {demoThreshold}. Low-quality evidence always abstains.</p>
                    </div>
                  </div>
                </section>

                <section className="grid gap-5 lg:grid-cols-[minmax(0,1.1fr)_minmax(340px,0.9fr)]">
                  <div className="panel overflow-hidden">
                    <div className="flex items-center justify-between border-b border-border px-5 py-4">
                      <div><p className="text-sm font-medium">Verified evidence</p><p className="mt-0.5 text-xs text-muted-foreground">Every finding is bound to a source</p></div><LockKeyhole className="size-4 text-muted-foreground" />
                    </div>
                    <Table>
                      <TableHeader><TableRow><TableHead>Signal</TableHead><TableHead>Result</TableHead><TableHead>Source</TableHead></TableRow></TableHeader>
                      <TableBody>
                        {selected.evidence.map((signal) => (
                          <TableRow key={signal.label}><TableCell className="font-medium">{signal.label}</TableCell><TableCell><SignalBadge strength={signal.strength}>{signal.value}</SignalBadge></TableCell><TableCell className="font-mono text-xs text-muted-foreground">{signal.source}</TableCell></TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>

                  <div className="panel p-5">
                    <div className="flex items-center gap-2">
                      <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10 text-primary"><Sparkles className="size-4" /></div>
                      <div><p className="text-sm font-medium">Grounded investigator</p><p className="text-xs text-muted-foreground">Structured output · no autonomous rejection</p></div>
                    </div>
                    <div className="mt-5 rounded-xl border border-border bg-muted/20 p-4">
                      <p className="text-sm leading-6">
                        {result.decision === "recapture" ? "The current images are insufficient for a reliable authenticity decision. Request a clear rear-label image before continuing." : result.decision === "review" ? "The returned item has multiple independently verified mismatches. Hold the automated refund and send this case to a human reviewer." : "The returned item is consistent with the dispatch fingerprint. No material mismatch is supported by the available evidence."}
                      </p>
                      <ul className="mt-4 space-y-2">
                        {result.reasons.slice(0, 3).map((reason) => <li key={reason} className="flex gap-2 text-xs leading-5 text-muted-foreground"><CircleDot className="mt-1 size-3 shrink-0 text-primary" />{reason}</li>)}
                      </ul>
                    </div>
                    <div className="mt-4 flex flex-wrap gap-2"><Badge variant="outline">order_record</Badge><Badge variant="outline">evidence_pair</Badge><Badge variant="outline">policy §4.2</Badge></div>
                    <div className="mt-5 grid grid-cols-3 gap-2">
                      <Button variant={result.decision === "approve" ? "default" : "outline"} size="sm" onClick={() => setRecordedDecision("approve")}>Approve</Button>
                      <Button variant={result.decision === "recapture" ? "default" : "outline"} size="sm" onClick={() => setRecordedDecision("recapture")}>Recapture</Button>
                      <Button variant={result.decision === "review" ? "default" : "outline"} size="sm" onClick={() => setRecordedDecision("review")}>Review</Button>
                    </div>
                    {recordedDecision && <p className="mt-3 flex items-center gap-2 text-xs text-primary"><Check className="size-3" /> Decision recorded for this browser session.</p>}
                  </div>
                </section>
              </div>
            </section>
          </TabsContent>

          <TabsContent value="live">
            <LiveVerification />
          </TabsContent>

          <TabsContent value="evaluation">
            <section className="space-y-5">
              <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-end">
                <div><p className="eyebrow">Locked benchmark / ABO visual substitution</p><h3 className="mt-2 text-xl font-semibold tracking-tight">Real models. Unseen product identities. Honest misses.</h3></div>
                <Badge variant="outline" className="w-fit border-primary/30 text-primary">Item-disjoint test · threshold locked at 0.92</Badge>
              </div>

              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <Metric label="Precision" value={`${(lockedTest.precision * 100).toFixed(1)}%`} note="95% CI 100–100%" />
                <Metric label="Recall" value={`${(lockedTest.recall * 100).toFixed(1)}%`} note="95% CI 66.7–100%" />
                <Metric label="F1 score" value={lockedTest.f1.toFixed(3)} note="95% CI 0.800–1.000" />
                <Metric label="Locked test pairs" value={lockedTest.cases.toString()} note="15 genuine · 15 substitutions" />
              </div>

              <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_360px]">
                <div className="space-y-5">
                  <div className="panel p-5 lg:p-6">
                    <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                      <div><p className="text-sm font-medium">Outcome audit</p><p className="mt-1 text-xs text-muted-foreground">No false review flags; two substitutions were missed.</p></div>
                      <Badge variant="outline">PR average precision · 0.996</Badge>
                    </div>
                    <div className="mt-6 grid gap-5 lg:grid-cols-[minmax(0,1fr)_260px]">
                      <div className="grid grid-cols-2 gap-3">
                        <ConfusionCell label="True substitution" value={lockedTest.confusion.tp} note="Detected" tone="safe" />
                        <ConfusionCell label="Missed substitution" value={lockedTest.confusion.fn} note="False negative" tone="risk" />
                        <ConfusionCell label="Genuine cleared" value={lockedTest.confusion.tn} note="True negative" tone="safe" />
                        <ConfusionCell label="False review flag" value={lockedTest.confusion.fp} note="False positive" tone="neutral" />
                      </div>
                      <div className="grid grid-cols-2 gap-3 lg:grid-cols-1">
                        <CostStat label="False-positive cost" value={money.format(lockedTest.false_positive_cost)} cost="0 genuine cases flagged" />
                        <CostStat label="Missed-swap loss" value={money.format(lockedTest.missed_substitution_loss)} cost="2 × ₹6,200" />
                      </div>
                    </div>
                    <div className="mt-5 grid gap-3 border-t border-border pt-5 sm:grid-cols-2 lg:grid-cols-4">
                      <MiniStat label="Calibration error" value={lockedTest.calibration_error.toFixed(3)} />
                      <MiniStat label="Recapture rate" value={`${(lockedTest.recapture_rate * 100).toFixed(1)}%`} />
                      <MiniStat label="Genuine recapture" value={`${(lockedTest.genuine_recapture_rate * 100).toFixed(1)}%`} />
                      <MiniStat label="Latency p50 / p95" value="11.5s / 65.4s" />
                    </div>
                  </div>

                  <div className="panel overflow-hidden">
                    <div className="border-b border-border px-5 py-4"><p className="text-sm font-medium">Category slices</p><p className="mt-0.5 text-xs text-muted-foreground">The lamp slice exposes both test misses; small slices should not be overgeneralised.</p></div>
                    <Table>
                      <TableHeader><TableRow><TableHead>Category</TableHead><TableHead>Cases</TableHead><TableHead>TP / FN</TableHead><TableHead>Precision</TableHead><TableHead>Recall</TableHead><TableHead>F1</TableHead></TableRow></TableHeader>
                      <TableBody>
                        <CategoryRow name="Chair" result={lockedTest.slices["category:chair"]} />
                        <CategoryRow name="Lamp" result={lockedTest.slices["category:lamp"]} warning />
                        <CategoryRow name="Sofa" result={lockedTest.slices["category:sofa"]} />
                        <CategoryRow name="Table" result={lockedTest.slices["category:table"]} />
                      </TableBody>
                    </Table>
                  </div>

                  <div className="panel p-5 lg:p-6">
                    <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
                      <div><p className="text-sm font-medium">POC threshold sandbox</p><p className="mt-1 text-xs text-muted-foreground">Explore cost sensitivity on 480 generated pairs. This never changes the locked result above.</p></div>
                      <Badge variant="outline" className="border-amber-400/30 text-amber-300">Synthetic · clearly separated</Badge>
                    </div>
                    <div className="mt-7 grid gap-8 lg:grid-cols-[minmax(0,1fr)_260px]">
                      <div>
                        <div className="flex items-end justify-between"><div><p className="text-xs uppercase tracking-wider text-muted-foreground">Demo threshold</p><p className="mt-1 text-3xl font-semibold tabular-nums">{demoThreshold}</p></div><p className="text-right text-xs leading-5 text-muted-foreground">Precision {(syntheticEvaluation.precision * 100).toFixed(1)}%<br />Recall {(syntheticEvaluation.recall * 100).toFixed(1)}%</p></div>
                        <Slider className="mt-6" min={40} max={90} step={1} value={[demoThreshold]} onValueChange={(value) => setDemoThreshold(value[0])} />
                        <div className="mt-3 flex justify-between text-[10px] uppercase tracking-wider text-muted-foreground"><span>More recall</span><span>Less friction</span></div>
                      </div>
                      <div className="grid grid-cols-2 gap-3"><CostStat label="False positives" value={syntheticEvaluation.falsePositives.toString()} cost={money.format(syntheticEvaluation.falsePositiveCost)} /><CostStat label="Missed swaps" value={syntheticEvaluation.falseNegatives.toString()} cost={money.format(syntheticEvaluation.missedLoss)} /></div>
                    </div>
                  </div>
                </div>

                <aside className="space-y-5">
                  <div className="panel p-5">
                    <div className="flex items-center gap-3"><div className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary"><FileCheck2 className="size-5" /></div><div><p className="text-sm font-medium">Reproducibility gate</p><p className="text-xs text-muted-foreground">Protocol item-disjoint-v1</p></div></div>
                    <div className="mt-6 space-y-4">
                      <Gate icon={CheckCircle2} title="Threshold locked" detail="0.92 selected on validation only" />
                      <Gate icon={CheckCircle2} title="Test exclusions disclosed" detail="2 prior smoke pairs removed" />
                      <Gate icon={CheckCircle2} title="Cost declared" detail="₹80 FP · ₹6,200 FN" />
                      <Gate icon={CheckCircle2} title="Model card complete" detail="Limits and intended use documented" />
                    </div>
                  </div>
                  <div className="panel p-5">
                    <p className="text-sm font-medium">Known limitations</p>
                    <ul className="mt-4 space-y-3 text-xs leading-5 text-muted-foreground">
                      <li className="flex gap-2"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-300" />Furniture-only ABO benchmark; not production merchant traffic.</li>
                      <li className="flex gap-2"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-300" />Lamp recall is 50%; both false negatives are in this slice.</li>
                      <li className="flex gap-2"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-300" />40% recapture and 65.4s p95 latency require improvement.</li>
                      <li className="flex gap-2"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-amber-300" />ABO does not validate serial-number or weight signals.</li>
                    </ul>
                  </div>
                  <div className="rounded-xl border border-primary/20 bg-primary/5 p-4"><p className="text-xs font-medium text-primary">Interpretation</p><p className="mt-2 text-xs leading-5 text-muted-foreground">The ranking is strong, but the wide recall interval and small category slices mean this is a credible POC result—not a production guarantee.</p></div>
                </aside>
              </div>
            </section>
          </TabsContent>

          <TabsContent value="pipeline">
            <section className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
              <div className="space-y-5">
                <div className="panel p-5 lg:p-6">
                  <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start">
                    <div>
                      <p className="eyebrow">Local RTX inference</p>
                      <h3 className="mt-2 text-xl font-semibold tracking-tight">Evidence engines. Calibrated fusion. Independent audit.</h3>
                      <p className="mt-2 max-w-2xl text-sm leading-6 text-muted-foreground">DINOv2 measures visual identity, Qwen3-VL extracts observable discrepancies, and a trained fusion model creates the recommendation. A separate API-based LLM audits consistency without changing the score or owning the final action.</p>
                    </div>
                    <Badge variant="outline" className="border-primary/30 text-primary">RTX 5050 smoke test verified</Badge>
                  </div>

                  <div className="mt-7 grid gap-3 md:grid-cols-2">
                    <PipelineStage icon={Cpu} name="DINOv2 small" role="Pair embedding similarity" runtime="Local CUDA · verified" ready />
                    <PipelineStage icon={Sparkles} name="Qwen3-VL 4B" role="Evidence-only comparison" runtime="4-bit · token-capped" ready />
                    <PipelineStage icon={Braces} name="Schema + grounder" role="Reject extra fields and citations" runtime="Unit-tested" ready />
                    <PipelineStage icon={ShieldCheck} name="Risk policy" role="Approve, recapture, or review" runtime="Deterministic" ready />
                    <PipelineStage icon={Activity} name="LLM evidence auditor" role="Independent consistency check" runtime="Optional API · advisory only" ready />
                  </div>
                </div>

                <div className="panel overflow-hidden">
                  <div className="flex items-center justify-between border-b border-border px-5 py-4"><div><p className="text-sm font-medium">Implementation truth table</p><p className="mt-0.5 text-xs text-muted-foreground">What is verified in code, hardware and the locked benchmark</p></div><LockKeyhole className="size-4 text-muted-foreground" /></div>
                  <Table>
                    <TableHeader><TableRow><TableHead>Layer</TableHead><TableHead>Status</TableHead><TableHead>Evidence</TableHead></TableRow></TableHeader>
                    <TableBody>
                      <ModelRow layer="Risk + abstention" status="Verified" evidence="Calibrated threshold plus explicit recapture gate" ready />
                      <ModelRow layer="Structured VLM guard" status="Verified" evidence="Schema, grounding and OOM retry tests" ready />
                      <ModelRow layer="Evidence auditor" status="Verified" evidence="Strict schema · API fallback · no score authority" ready />
                      <ModelRow layer="Safety test suite" status="Verified" evidence="41 local tests passing" ready />
                      <ModelRow layer="Local API" status="Implemented" evidence="POST /v1/verify · lazy GPU load" ready />
                      <ModelRow layer="Model execution" status="Verified" evidence="Real mouse pair · RTX 5050 · 4-bit VLM" ready />
                      <ModelRow layer="Benchmark harness" status="Verified" evidence="Leakage, cost, calibration, CI and slices" ready />
                      <ModelRow layer="ABO dataset builder" status="Verified" evidence="Bounded download · deterministic item splits" ready />
                      <ModelRow layer="Fusion trainer" status="Verified" evidence="OOF calibration · JSON model · test isolation" ready />
                      <ModelRow layer="Real-image benchmark" status="Verified" evidence="30 locked pairs · 100% precision · 86.7% recall" ready />
                    </TableBody>
                  </Table>
                </div>
              </div>

              <aside className="panel h-fit p-5">
                <div className="flex items-center gap-3"><div className="flex size-10 items-center justify-center rounded-xl bg-primary/10 text-primary"><ShieldCheck className="size-5" /></div><div><p className="text-sm font-medium">Authority boundary</p><p className="text-xs text-muted-foreground">Enforced in code, not prompt alone</p></div></div>
                <div className="mt-6 space-y-4">
                  <Gate icon={CheckCircle2} title="No fraud accusation" detail="Only observable mismatches are allowed" />
                  <Gate icon={CheckCircle2} title="No autonomous rejection" detail="Adverse outcomes require a reviewer" />
                  <Gate icon={CheckCircle2} title="Prompt injection contained" detail="Pixels and OCR remain untrusted data" />
                  <Gate icon={CheckCircle2} title="Unsupported claims removed" detail="Every observation cites available evidence" />
                </div>
                <div className="mt-6 rounded-xl border border-primary/20 bg-primary/5 p-4"><p className="text-xs font-medium text-primary">Verified smoke result</p><p className="mt-2 text-xs leading-5 text-muted-foreground">The first real mouse pair produced 95% same-product likelihood. The policy still requested recapture because both photos fell below the sharpness gate—an intentional abstention, not a fraud decision.</p></div>
              </aside>
            </section>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}

function EvidenceCard({ title, subtitle, sku, serial, weight, variant }: { title: string; subtitle: string; sku: string; serial: string | null; weight: number; variant: "dispatch" | "return" }) {
  return (
    <div className="overflow-hidden rounded-2xl border border-border bg-muted/20">
      <div className={`evidence-visual ${variant}`}><div className="scan-grid" /><div className="evidence-mark"><ScanLine className="size-7" /></div><span className="evidence-code">{variant === "dispatch" ? "OUTBOUND" : "RETURN"} / {sku}</span></div>
      <div className="p-4"><div className="flex items-start justify-between gap-3"><div><p className="text-sm font-medium">{title}</p><p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p></div><Badge variant="outline">{weight} g</Badge></div><div className="mt-3 flex items-center justify-between border-t border-border pt-3 text-xs"><span className="text-muted-foreground">Serial</span><span className="font-mono">{serial ?? "Not applicable"}</span></div></div>
    </div>
  )
}

function SignalBadge({ strength, children }: { strength: "match" | "mismatch" | "uncertain"; children: React.ReactNode }) { return <span className={`signal-badge ${strength}`}>{children}</span> }
function Metric({ label, value, note }: { label: string; value: string; note: string }) { return <div className="panel p-5"><p className="text-xs uppercase tracking-wider text-muted-foreground">{label}</p><p className="mt-3 text-3xl font-semibold tabular-nums">{value}</p><p className="mt-2 text-xs text-muted-foreground">{note}</p></div> }
function CostStat({ label, value, cost }: { label: string; value: string; cost: string }) { return <div className="rounded-xl border border-border bg-muted/20 p-3"><p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p><p className="mt-2 text-2xl font-semibold tabular-nums">{value}</p><p className="mt-1 text-xs text-amber-300">{cost}</p></div> }
function MiniStat({ label, value }: { label: string; value: string }) { return <div><p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p><p className="mt-1 text-sm font-medium tabular-nums">{value}</p></div> }
function ConfusionCell({ label, value, note, tone }: { label: string; value: number; note: string; tone: "safe" | "risk" | "neutral" }) { return <div className={`rounded-xl border p-4 ${tone === "risk" ? "border-red-400/25 bg-red-400/5" : tone === "safe" ? "border-primary/20 bg-primary/5" : "border-border bg-muted/20"}`}><p className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</p><p className={`mt-2 text-3xl font-semibold tabular-nums ${tone === "risk" ? "text-red-300" : tone === "safe" ? "text-primary" : ""}`}>{value}</p><p className="mt-1 text-xs text-muted-foreground">{note}</p></div> }
type CategoryResult = typeof realReport.test.slices["category:chair"]
function CategoryRow({ name, result, warning = false }: { name: string; result: CategoryResult; warning?: boolean }) { return <TableRow className={warning ? "bg-amber-400/5" : undefined}><TableCell className="font-medium">{name}{warning && <Badge variant="outline" className="ml-2 border-amber-400/30 text-amber-300">weak slice</Badge>}</TableCell><TableCell>{result.cases}</TableCell><TableCell className="font-mono text-xs">{result.confusion.tp} / {result.confusion.fn}</TableCell><TableCell>{(result.precision * 100).toFixed(0)}%</TableCell><TableCell className={warning ? "text-amber-300" : undefined}>{(result.recall * 100).toFixed(0)}%</TableCell><TableCell>{result.f1.toFixed(3)}</TableCell></TableRow> }
function Gate({ icon: Icon, title, detail, pending = false }: { icon: typeof CheckCircle2; title: string; detail: string; pending?: boolean }) { return <div className="flex gap-3"><Icon className={`mt-0.5 size-4 shrink-0 ${pending ? "text-muted-foreground" : "text-primary"}`} /><div><p className="text-sm">{title}</p><p className="mt-0.5 text-xs text-muted-foreground">{detail}</p></div></div> }
function PipelineStage({ icon: Icon, name, role, runtime, ready = false }: { icon: typeof Cpu; name: string; role: string; runtime: string; ready?: boolean }) { return <div className="rounded-xl border border-border bg-muted/20 p-4"><div className="flex items-start justify-between gap-3"><div className="flex size-9 items-center justify-center rounded-lg bg-primary/10 text-primary"><Icon className="size-4" /></div><StatusPill ready={ready} /></div><p className="mt-4 text-sm font-medium">{name}</p><p className="mt-1 text-xs text-muted-foreground">{role}</p><p className="mt-3 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{runtime}</p></div> }
function StatusPill({ ready, label }: { ready: boolean; label?: string }) { return <span className={`rounded-full border px-2 py-1 text-[10px] uppercase tracking-wider ${ready ? "border-primary/25 bg-primary/5 text-primary" : "border-amber-400/25 bg-amber-400/5 text-amber-300"}`}>{label ?? (ready ? "Ready" : "GPU pending")}</span> }
function ModelRow({ layer, status, evidence, ready = false }: { layer: string; status: string; evidence: string; ready?: boolean }) { return <TableRow><TableCell className="font-medium">{layer}</TableCell><TableCell><StatusPill ready={ready} label={status} /></TableCell><TableCell className="text-muted-foreground">{evidence}</TableCell></TableRow> }
