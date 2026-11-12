function resolveApiBase() {
    const explicitBase = String(window.CLARION_API_BASE || "").trim();
    if (explicitBase) {
        return explicitBase.replace(/\/+$/, "");
    }

    const { protocol, hostname } = window.location;
    if (protocol === "http:" || protocol === "https:") {
        const host = hostname || "localhost";
        return `${protocol}//${host}:8000`;
    }

    return "http://localhost:8000";
}

function formatFetchError(error, pathHint = "") {
    const rawMessage = String(error?.message || error || "").trim();
    const isNetworkError = /failed to fetch|networkerror|load failed/i.test(rawMessage);
    if (!isNetworkError) {
        return rawMessage || "Request failed";
    }

    const endpoint = `${API_BASE}${pathHint}`;
    if (window.location.protocol === "file:") {
        return `Cannot reach API (${endpoint}). Open this UI via http://localhost:8080 (not file://).`;
    }

    const origin = window.location.origin || "unknown origin";
    return `Cannot reach API (${endpoint}). Ensure backend is running on port 8000 and CORS allows ${origin}.`;
}

const API_BASE = resolveApiBase();
const WS_BASE = API_BASE.replace(/^http/i, "ws");
const AUTO_START_ANALYSIS_AFTER_UPLOAD = true;

const PIPELINE_STAGES = [
    { key: "ingestion", label: "Upload & Validation" },
    { key: "chunking", label: "Semantic Chunking" },
    { key: "embedding", label: "Vector Embedding" },
    { key: "mapping", label: "Concept & Relation Mapping" },
    { key: "graph_building", label: "Knowledge Graph" },
    { key: "evaluation", label: "Grounding & Verification" },
    { key: "hierarchy", label: "Structural Hierarchy" },
];
const STAGE_INDEX = Object.fromEntries(PIPELINE_STAGES.map((stage, index) => [stage.key, index]));

let selectedDocId = null;
let selectedDoc = null;
let currentDocuments = [];
let graph = null;
let logSocket = null;
let monitorTimer = null;
let logPollTimer = null;
let currentKnowledgeMap = null;
let currentSummary = null;

document.addEventListener("DOMContentLoaded", () => {
    initPipelineBoard();
    initGraph();
    bindEvents();
    connectLogStream();
    loadDatasetStats();
    if (window.location.protocol === "file:") {
        setStatus("upload-status", "Serve the frontend on http://localhost:8080. file:// blocks API uploads.", "error");
    }
    loadDocuments();
    refreshSystemStatus();
    setInterval(refreshSystemStatus, 5000);
});

function bindEvents() {
    document.getElementById("upload-form").addEventListener("submit", uploadDocument);
    document.getElementById("refresh-btn").addEventListener("click", loadDocuments);
    document.getElementById("analyze-btn").addEventListener("click", startAnalysis);
    document.getElementById("clear-logs-btn").addEventListener("click", () => {
        document.getElementById("logs-console").textContent = "";
    });
    document.getElementById("refresh-report-btn").addEventListener("click", loadAnalysisReport);

    const layoutSelect = document.getElementById("graph-layout");
    if (layoutSelect) {
        layoutSelect.addEventListener("change", (e) => {
            applyGraphLayout(e.target.value);
        });
    }

    const resetFocusBtn = document.getElementById("reset-focus");
    if (resetFocusBtn) {
        resetFocusBtn.addEventListener("click", resetGraphFocus);
    }

    document.getElementById("zoom-in").addEventListener("click", () => {
        if (graph) graph.zoom(graph.zoom() * 1.25);
    });
    document.getElementById("zoom-out").addEventListener("click", () => {
        if (graph) graph.zoom(graph.zoom() * 0.8);
    });
    document.getElementById("fit-graph").addEventListener("click", () => {
        if (graph) graph.fit(undefined, 35);
    });
    ["graph-search", "graph-node-filter", "graph-edge-filter"].forEach((id) => {
        document.getElementById(id)?.addEventListener("input", applyGraphFilters);
        document.getElementById(id)?.addEventListener("change", applyGraphFilters);
    });

    // Grounded RAG Query events
    const ragBtn = document.getElementById("rag-query-btn");
    const ragInput = document.getElementById("rag-query-input");
    if (ragBtn && ragInput) {
        ragBtn.addEventListener("click", askDocument);
        ragInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") askDocument();
        });
    }

    const copyRagBtn = document.getElementById("copy-rag-btn");
    if (copyRagBtn) {
        copyRagBtn.addEventListener("click", copyRagAnswer);
    }

    // Quick prompt chips
    document.querySelectorAll(".prompt-chip").forEach((chip) => {
        chip.addEventListener("click", () => {
            const query = chip.dataset.query;
            if (ragInput && query) {
                ragInput.value = query;
                askDocument();
            }
        });
    });
}

async function uploadDocument(event) {
    event.preventDefault();
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];
    if (!file) {
        setStatus("upload-status", "Select a PDF or DOCX file first.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    setStatus("upload-status", "Uploading document...", "info");
    try {
        const response = await fetch(`${API_BASE}/upload`, {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Upload failed");
        }
        setStatus("upload-status", `Uploaded: ${payload.filename}`, "success");
        fileInput.value = "";
        const documents = await loadDocuments();
        const uploadedDoc = (documents || []).find((doc) => doc.id === payload.document_id);
        if (uploadedDoc) {
            selectDocument(uploadedDoc);
            if (AUTO_START_ANALYSIS_AFTER_UPLOAD) {
                setStatus("analysis-status", "Auto-starting analysis for uploaded document...", "info");
                await startAnalysis();
            }
        }
    } catch (error) {
        setStatus("upload-status", `Upload error: ${formatFetchError(error, "/upload")}`, "error");
    }
}

async function loadDocuments() {
    const container = document.getElementById("documents-list");
    container.innerHTML = "";
    try {
        const response = await fetch(`${API_BASE}/upload/list`);
        const payload = await response.json();
        const documents = payload.documents || [];
        currentDocuments = documents;
        if (!documents.length) {
            container.innerHTML = `<div class="muted">No documents uploaded.</div>`;
            return [];
        }

        for (const doc of documents) {
            if (selectedDocId === doc.id) {
                selectedDoc = doc;
            }
            const item = document.createElement("button");
            item.type = "button";
            item.className = "document-item";
            item.dataset.docId = doc.id;
            item.dataset.id = doc.id;
            if (selectedDocId === doc.id) {
                item.classList.add("selected");
            }
            const sizeMb = doc.size ? `${(doc.size / (1024 * 1024)).toFixed(2)} MB` : "Unknown size";
            const statusLabel = String(doc.status || "uploaded").toUpperCase();
            item.innerHTML = `
                <span class="name">${escapeHtml(doc.name || "Untitled")}</span>
                <span class="meta">${escapeHtml(statusLabel)} • ${sizeMb} • ${doc.pages ? doc.pages + " pages" : "DOCX"}</span>
            `;
            item.addEventListener("click", () => selectDocument(doc));
            container.appendChild(item);
        }

        return documents;
    } catch (error) {
        container.innerHTML = `<div class="muted">${escapeHtml(formatFetchError(error, "/upload/list"))}</div>`;
        return [];
    }
}

function selectDocument(docOrId) {
    if (!docOrId) return;
    let doc = docOrId;
    if (typeof docOrId === "string") {
        doc = (currentDocuments || []).find((d) => d && (d.id === docOrId || d.document_id === docOrId)) || {
            id: docOrId,
            name: docOrId,
            status: "uploaded",
        };
    }
    if (!doc || !doc.id) return;

    selectedDocId = doc.id;
    selectedDoc = doc;
    document.querySelectorAll(".document-item").forEach((item) => item.classList.remove("selected"));
    const matching = document.querySelector(`.document-item[data-doc-id="${selectedDocId}"], .document-item[data-id="${selectedDocId}"]`);
    if (matching) matching.classList.add("selected");

    const sizeFormatted = doc.size ? (doc.size / 1024 / 1024).toFixed(2) : "0.00";
    const docPages = doc.page_count ?? doc.pages;
    const docInfo = document.getElementById("doc-info");
    if (docInfo) {
        docInfo.innerHTML = `
            <strong>${escapeHtml(doc.name || "Untitled")}</strong><br>
            Status: <span class="badge">${escapeHtml(doc.status || 'uploaded')}</span><br>
            Size: ${sizeFormatted} MB • Physical pages: ${docPages ?? 'Unavailable'} • Normalized words: ${doc.word_count !== null && doc.word_count !== undefined ? Number(doc.word_count).toLocaleString() : 'Unavailable'}
        `;
    }
    loadDocumentMetrics(doc.id);

    const statusDocName = document.getElementById("status-doc-name");
    if (statusDocName) statusDocName.textContent = doc.name || "Untitled";

    const overviewTitle = document.getElementById("overview-doc-title");
    if (overviewTitle) overviewTitle.textContent = `${doc.name || "Untitled"} (${doc.id})`;

    updateOverviewMetric("stat-words", doc.word_count ?? "--");
    updateOverviewMetric("stat-pages", docPages ?? "--");
    ["sections", "chunks", "concepts", "relations", "nodes", "edges"].forEach((m) => {
        const el = document.getElementById(`stat-${m}`);
        if (el) el.textContent = "--";
    });

    const analyzeBtn = document.getElementById("analyze-btn");
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = isDocumentAnalyzed(doc) ? "Re-run Full Analysis" : "Start Full Analysis";
    }

    const ragBtn = document.getElementById("rag-query-btn");
    if (ragBtn) ragBtn.disabled = !isDocumentAnalyzed(doc);

    resetPipelineBoard();
    connectLogStream();

    if (isDocumentAnalyzed(doc)) {
        loadGraph(doc.id);
        loadAnalysisReport();
        loadDocumentSummary(doc.id);
        loadKnowledgeMap(doc.id);
    } else {
        resetWorkspaceViews("Document pending analysis. Click 'Start Full Analysis' to begin.");
    }

    refreshPipelineStatus();
    monitorPipeline();
}

window.selectDocument = selectDocument;

function resetWorkspaceViews(message) {
    resetGraphPanels(message);
    const summaryBox = document.getElementById("overall-summary-box");
    if (summaryBox) summaryBox.textContent = message;

    const insightsList = document.getElementById("key-insights-list");
    if (insightsList) insightsList.innerHTML = `<li class="muted">${escapeHtml(message)}</li>`;

    const typeBadge = document.getElementById("doc-type-badge");
    if (typeBadge) typeBadge.textContent = "Type: Pending";

    const modeBadge = document.getElementById("insights-mode-badge");
    if (modeBadge) {
        modeBadge.textContent = "Analysis: Pending";
        modeBadge.className = "mode-badge";
    }

    const reportEl = document.getElementById("analysis-report");
    if (reportEl) reportEl.textContent = message;

    const structContainer = document.getElementById("structure-container");
    if (structContainer) structContainer.innerHTML = `<div class="muted">${escapeHtml(message)}</div>`;

    const structCount = document.getElementById("structure-count");
    if (structCount) structCount.textContent = "0 sections";

    // Reset Top Extracted Concepts table and counter
    renderConceptsTable([]);

    // Reset Selected Node Details Drawer / Inspector
    const inspectorTitle = document.getElementById("inspector-title");
    if (inspectorTitle) inspectorTitle.textContent = "Concept Inspector";
    const inspectorBadge = document.getElementById("inspector-badge");
    if (inspectorBadge) inspectorBadge.textContent = "Click a node to inspect";
    const inspectorBody = document.getElementById("inspector-body");
    if (inspectorBody) {
        inspectorBody.innerHTML = `<p class="muted">Select any concept node in the knowledge map above to view its grounded textual definition, occurrence frequency, supporting chunks, and candidate relations.</p>`;
    }

    // Reset RAG query output and status
    const ragBox = document.getElementById("rag-output-box");
    if (ragBox) ragBox.style.display = "none";
    const ragAnswer = document.getElementById("rag-answer-text");
    if (ragAnswer) ragAnswer.textContent = "";
    const ragSources = document.getElementById("rag-sources-list");
    if (ragSources) ragSources.innerHTML = "";
    const ragStatus = document.getElementById("rag-status");
    if (ragStatus) {
        ragStatus.textContent = "";
        ragStatus.className = "status-box";
    }

    // Clear cached knowledge map and summary objects
    currentKnowledgeMap = null;
    currentSummary = null;

    // Keep known upload metrics (words, pages) if available; reset stage-specific analysis metrics
    const words = selectedDoc?.word_count;
    const pages = selectedDoc?.page_count ?? selectedDoc?.pages;
    updateOverviewMetric("stat-words", words !== undefined && words !== null ? words : "--");
    updateOverviewMetric("stat-pages", pages !== undefined && pages !== null ? pages : "--");
    ["sections", "chunks", "concepts", "relations", "nodes", "edges"].forEach((m) => {
        const el = document.getElementById(`stat-${m}`);
        if (el) el.textContent = "--";
    });

    const relPanel = document.getElementById("candidate-relations-panel");
    if (relPanel) relPanel.style.display = "none";
    const relCount = document.getElementById("relations-count");
    if (relCount) relCount.textContent = "0 relations";
    const relTbody = document.getElementById("relations-table-body");
    if (relTbody) relTbody.innerHTML = `<tr><td colspan="5" class="muted text-center">No candidate relations loaded</td></tr>`;

    loadDatasetStats();
}

async function loadDocumentMetrics(documentId) {
    const panel = document.getElementById("doc-processing-stats");
    if (!panel) return;
    panel.textContent = "Loading extraction and indexing counts…";
    try {
        const response = await fetch(`${API_BASE}/upload/${encodeURIComponent(documentId)}/metrics`);
        if (!response.ok) throw new Error(`Metrics unavailable (${response.status})`);
        const payload = await response.json();
        if (selectedDocId !== documentId) return;
        const c = payload.counts || {};
        const show = (value) => value === null || value === undefined ? "Not recorded" : Number(value).toLocaleString();
        panel.innerHTML = `
            <div><strong>Pages</strong> Physical ${show(c.physical_pages)} · extracted ${show(c.extracted_pages)} · with text ${show(c.text_bearing_pages)} · indexed ${show(c.indexed_pages)}</div>
            <div><strong>Words</strong> raw ${show(c.raw_words)} · normalized ${show(c.normalized_words)}</div>
            <div><strong>Chunks</strong> raw candidates ${show(c.candidate_raw_chunks)} · retained ${show(c.retained_chunks)} · rejected ${show(c.rejected_chunks)}</div>
            <div><strong>Embedding</strong> segments ${show(c.embedding_segments)} · mapped vectors ${show(c.mapped_vectors)} · FAISS vectors ${show(c.faiss_vectors)}</div>
            <small>${escapeHtml(payload.word_count_definition || "")}</small>
        `;
    } catch (error) {
        if (selectedDocId === documentId) panel.textContent = `Processing counts unavailable: ${error.message}`;
    }
}

async function startAnalysis() {
    if (!selectedDocId) return;
    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing Pipeline...";
    setStatus("analysis-status", "Submitting pipeline job to background orchestrator...", "info");
    document.getElementById("analysis-report").textContent = "Analysis in progress. Synthesizing structure, embeddings, knowledge map, and evaluation...";
    resetPipelineBoard();

    try {
        const response = await fetch(`${API_BASE}/analyze/${selectedDocId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                generate_hierarchy: true,
                run_evaluation: true,
                force_reanalyze: true,
            }),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Failed to start analysis");
        }
        setStatus("analysis-status", "Job queued. Monitoring pipeline execution...", "info");
        monitorPipeline();
    } catch (error) {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Start Full Analysis";
        setStatus("analysis-status", `Job submission error: ${formatFetchError(error, "/analyze")}`, "error");
    }
}

function monitorPipeline() {
    if (monitorTimer) {
        clearInterval(monitorTimer);
    }
    monitorTimer = setInterval(async () => {
        if (!selectedDocId) return;
        await refreshPipelineStatus();
        await loadAnalysisReport();
    }, 1500);
}

async function refreshPipelineStatus() {
    if (!selectedDocId) return;
    try {
        const response = await fetch(
            `${API_BASE}/system/pipeline-status?document_id=${encodeURIComponent(selectedDocId)}`
        );
        if (response.ok) {
            const payload = await response.json();
            if (payload && payload.status !== "not_found") {
                renderPipelineStatus(payload);

                const summary = document.getElementById("pipeline-summary");
                const modeChip = document.getElementById("status-mode");
                const modelChip = document.getElementById("status-model");
                const durationChip = document.getElementById("status-duration");

                const isDegraded = payload.status === "degraded" || payload.analysis_mode === "deterministic_fallback";
                const statusDisplay = isDegraded ? "Degraded (Fallback)" : payload.status;
                summary.textContent = `${statusDisplay} — ${payload.overall_progress || 0}%`;

                if (payload.analysis_mode === "deterministic_fallback") {
                    modeChip.textContent = "Deterministic Fallback";
                    document.getElementById("mode-chip").className = "status-chip mode-fallback";
                    updateHonestStatus("Fallback Active", "fallback");
                } else if (payload.analysis_mode === "llm") {
                    modeChip.textContent = "LLM Grounded";
                    document.getElementById("mode-chip").className = "status-chip";
                    updateHonestStatus("LLM Ready", "ready");
                }

                if (payload.active_model) {
                    modelChip.textContent = payload.active_model;
                }

                const durVal = payload.total_duration_sec ?? payload.duration_seconds;
                if (durVal !== undefined && durVal !== null) {
                    durationChip.textContent = `${Number(durVal).toFixed(1)}s`;
                }

                if (payload.status === "completed" || payload.status === "degraded" || payload.status === "failed") {
                    const analyzeBtn = document.getElementById("analyze-btn");
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = "Re-run Full Analysis";

                    const ragBtn = document.getElementById("rag-query-btn");
                    if (ragBtn) ragBtn.disabled = false;

                    if (payload.status === "completed" || payload.status === "degraded") {
                        const statusMsg = isDegraded
                            ? "Pipeline completed with deterministic fallback analysis."
                            : "Pipeline completed successfully.";
                        setStatus("analysis-status", statusMsg, isDegraded ? "info" : "success");
                        await loadDocuments();
                        await loadGraph(selectedDocId);
                        await loadDocumentSummary(selectedDocId);
                        await loadKnowledgeMap(selectedDocId);
                    }
                    if (payload.status === "failed") {
                        updateHonestStatus("Failed", "failed");
                        setStatus("analysis-status", `Pipeline failed: ${payload.error || "unknown error"}`, "error");
                    }
                    clearInterval(monitorTimer);
                    monitorTimer = null;
                }
            } else if (payload && payload.status === "not_found") {
                if (isDocumentAnalyzed(selectedDoc)) {
                    // Analyzed document disconnect recovery: query persisted analysis report
                    try {
                        const reportRes = await fetch(`${API_BASE}/system/analysis-report/${encodeURIComponent(selectedDocId)}`);
                        if (reportRes.ok) {
                            const rep = await reportRes.json();
                            if (rep && (rep.summary || rep.metrics || rep.stage_validations || rep.artifacts)) {
                                const durVal = rep.total_duration_sec ?? (
                                    rep.started_at && rep.completed_at
                                        ? Math.max(0, (new Date(rep.completed_at).getTime() - new Date(rep.started_at).getTime()) / 1000)
                                        : null
                                );

                                const stageMap = {};
                                const stagesList = rep.stages_completed && rep.stages_completed.length > 0
                                    ? rep.stages_completed
                                    : ["ingestion", "chunking", "embedding", "mapping", "graph_building"];
                                stagesList.forEach((s) => {
                                    stageMap[s] = { status: "completed", duration_sec: null };
                                });
                                if (rep.stage_validations && Array.isArray(rep.stage_validations)) {
                                    rep.stage_validations.forEach((sv) => {
                                        stageMap[sv.stage] = {
                                            status: sv.status || (sv.validation_passed ? "completed" : "completed"),
                                            duration_sec: (sv.quality_metrics && sv.quality_metrics.duration_sec) ?? null,
                                        };
                                    });
                                }

                                const isDegraded = rep.status === "degraded" || rep.analysis_mode === "deterministic_fallback";
                                const syntheticPayload = {
                                    document_id: selectedDocId,
                                    status: rep.status || (rep.success ? "completed" : "failed"),
                                    analysis_mode: rep.analysis_mode || (isDegraded ? "deterministic_fallback" : "llm"),
                                    active_model: rep.active_model || "qwen3.5:4b",
                                    total_duration_sec: durVal,
                                    overall_progress: 100,
                                    stages: stageMap,
                                };

                                renderPipelineStatus(syntheticPayload);
                                renderAnalysisReport(rep);

                                const summary = document.getElementById("pipeline-summary");
                                const modeChip = document.getElementById("status-mode");
                                const modelChip = document.getElementById("status-model");
                                const durationChip = document.getElementById("status-duration");

                                summary.textContent = `${isDegraded ? "Degraded (Fallback)" : "Completed"} — 100%`;
                                if (isDegraded) {
                                    modeChip.textContent = "Deterministic Fallback";
                                    document.getElementById("mode-chip").className = "status-chip mode-fallback";
                                    updateHonestStatus("Fallback Active", "fallback");
                                } else {
                                    modeChip.textContent = "LLM Grounded";
                                    document.getElementById("mode-chip").className = "status-chip";
                                    updateHonestStatus("LLM Ready", "ready");
                                }

                                if (rep.active_model) modelChip.textContent = rep.active_model;
                                if (durVal !== null && durVal !== undefined) {
                                    durationChip.textContent = `${Number(durVal).toFixed(1)}s`;
                                }

                                const analyzeBtn = document.getElementById("analyze-btn");
                                analyzeBtn.disabled = false;
                                analyzeBtn.textContent = "Re-run Full Analysis";

                                const ragBtn = document.getElementById("rag-query-btn");
                                if (ragBtn) ragBtn.disabled = false;
                            }
                        }
                    } catch (e) {
                        console.warn("Failed to recover from analysis report:", e);
                    }
                } else {
                    // Document is pending analysis: maintain truthful pending state
                    const summary = document.getElementById("pipeline-summary");
                    if (summary) summary.textContent = "Pending Analysis — 0%";
                    const timelineBody = document.getElementById("timeline-table-body");
                    if (timelineBody) {
                        timelineBody.innerHTML = `<tr><td colspan="5" class="muted text-center">Pipeline has not executed. Click 'Start Full Analysis' to begin.</td></tr>`;
                    }
                }
            }
        }
    } catch (error) {
        // Silent poll error
    }
}

async function loadDocumentSummary(docId) {
    try {
        const response = await fetch(`${API_BASE}/summary/${encodeURIComponent(docId)}`);
        if (!response.ok) return;
        const summary = await response.json();
        currentSummary = summary;
        renderSummaryAndInsights(summary);
        renderDocumentStructure(summary.sections || []);
        updateOverviewMetric("stat-sections", (summary.sections || []).length);
        if (summary.metadata && summary.metadata.chunk_count !== undefined && summary.metadata.chunk_count !== null) {
            updateOverviewMetric("stat-chunks", summary.metadata.chunk_count);
        }
    } catch (err) {
        console.error("Failed to load document summary", err);
    }
}

function isGroundedInsight(text) {
    if (!text || typeof text !== "string") return false;
    const trimmed = text.trim();
    if (trimmed.length < 15) return false;
    // Strictly filter out synthetic template artifacts
    if (/^Covers\s+[a-z0-9_,\s]+[.!?]?$/i.test(trimmed)) return false;
    if (/^Also touches on\s+[a-z0-9_,\s]+[.!?]?$/i.test(trimmed)) return false;
    if (trimmed.startsWith("Deterministic fallback analysis")) return false;
    if (trimmed.startsWith("LLM unavailable. Running deterministic fallback")) return false;
    return true;
}

function renderSummaryAndInsights(summary) {
    const typeBadge = document.getElementById("doc-type-badge");
    const modeBadge = document.getElementById("insights-mode-badge");
    const overallBox = document.getElementById("overall-summary-box");
    const insightsList = document.getElementById("key-insights-list");

    const docType = summary.document_type || "unknown";
    typeBadge.textContent = `Type: ${docType.replace("_", " ")}`;

    const isFallback = Boolean(
        summary.metadata?.fallback_used ||
        summary.metadata?.analysis_mode === "deterministic_fallback" ||
        summary.metadata?.llm_summary_accepted === false
    );
    if (isFallback) {
        modeBadge.textContent = "Mode: Deterministic Fallback";
        modeBadge.className = "mode-badge";
    } else {
        modeBadge.textContent = "Mode: LLM Grounded";
        modeBadge.className = "mode-badge llm-mode";
    }

    overallBox.textContent = summary.overall_summary || "No synthesized summary available.";

    // Generate grounded bullet points from verified sections and summary
    const bullets = [];
    if (summary.overall_summary) {
        const sentences = summary.overall_summary
            .split(/(?<=[.!?])\s+/)
            .map(s => s.trim())
            .filter(isGroundedInsight);
        for (const s of sentences) {
            if (bullets.length >= 2) break;
            bullets.push(s);
        }
    }

    const sections = summary.sections || [];
    for (const sec of sections) {
        if (bullets.length >= 5) break;
        const validPoints = (sec.key_points || []).map(p => p.trim()).filter(isGroundedInsight);
        if (validPoints.length > 0) {
            bullets.push(`${sec.title}: ${validPoints[0]}`);
        } else if (sec.summary && isGroundedInsight(sec.summary)) {
            bullets.push(`${sec.title}: ${sec.summary.trim()}`);
        }
    }

    if (bullets.length === 0) {
        if (sections.length > 0) {
            insightsList.innerHTML = `<li>Document structured into ${sections.length} analyzed sections across extracted body.</li>`;
        } else {
            insightsList.innerHTML = `<li class="muted">No key structural insights extracted.</li>`;
        }
    } else {
        insightsList.innerHTML = bullets.map(b => `<li>${escapeHtml(b)}</li>`).join("");
    }
}

async function loadKnowledgeMap(docId) {
    try {
        const response = await fetch(`${API_BASE}/knowledge-map/${encodeURIComponent(docId)}`);
        if (!response.ok) return;
        const km = await response.json();
        currentKnowledgeMap = km;
        renderConceptsTable(km.concepts || []);
        renderCandidateRelationsTable(km.relations || []);
        updateOverviewMetric("stat-concepts", (km.concepts || []).length);
        updateOverviewMetric("stat-relations", (km.relations || []).length);
    } catch (err) {
        console.error("Failed to load knowledge map", err);
    }
}

function renderConceptsTable(concepts) {
    const tbody = document.querySelector("#concepts-table tbody");
    const countBadge = document.getElementById("concepts-count");
    tbody.innerHTML = "";

    if (!concepts || concepts.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="muted text-center">No concepts loaded</td></tr>`;
        countBadge.textContent = "0 concepts";
        return;
    }

    countBadge.textContent = `${concepts.length} concepts`;
    const sorted = [...concepts].sort((a, b) => (Number(b.frequency || 1) - Number(a.frequency || 1)));

    tbody.innerHTML = sorted.map((c) => {
        const sections = (c.supporting_sections || []).slice(0, 3).map(s => `<span class="badge">${escapeHtml(s)}</span>`).join(" ");
        const definition = c.definition || c.context || "No description is available from the extracted source context.";

        // Centrality calculation: degree and normalized centrality from knowledge map relations
        const cName = (c.name || "").toLowerCase();
        const degree = (currentKnowledgeMap?.relations || []).filter(r =>
            (r.from_concept || "").toLowerCase() === cName || (r.to_concept || "").toLowerCase() === cName
        ).length;
        const totalOthers = Math.max(1, concepts.length - 1);
        const normCentrality = (degree / totalOthers).toFixed(3);
        const centralityDisplay = c.centrality !== undefined ? Number(c.centrality).toFixed(3) : `${normCentrality} (deg ${degree})`;

        return `
            <tr>
                <td><strong>${escapeHtml(c.name)}</strong></td>
                <td><span class="badge">${Number(c.frequency || 1)} occurrences</span></td>
                <td><span class="confidence-score" title="Degree: ${degree}, Normalized: ${normCentrality}">${centralityDisplay}</span></td>
                <td>${sections || '<span class="muted">Document Body</span>'}</td>
                <td>${escapeHtml(definition)}</td>
            </tr>
        `;
    }).join("");
}

function renderCandidateRelationsTable(relations) {
    const panel = document.getElementById("candidate-relations-panel");
    const tbody = document.querySelector("#relations-table tbody");
    const countBadge = document.getElementById("relations-count");
    if (!tbody) return;
    tbody.innerHTML = "";

    if (!relations || relations.length === 0) {
        if (panel) panel.style.display = "none";
        if (countBadge) countBadge.textContent = "0 relations";
        return;
    }

    if (panel) panel.style.display = "block";
    if (countBadge) countBadge.textContent = `${relations.length} relations`;
    const sorted = [...relations].sort((a, b) => (Number(b.confidence || 0) - Number(a.confidence || 0)));

    tbody.innerHTML = sorted.map((r) => {
        const relType = r.relation_type || "explanation";
        const conf = Number(r.confidence || 0).toFixed(2);
        const evidence = r.evidence_text || "No supporting passage is attached to this candidate.";
        const chunkId = r.source_chunk_id || (r.source_chunk_ids && r.source_chunk_ids[0]) || (r.chunk_ids && r.chunk_ids[0]);
        const citation = chunkId ? `<div class="chunk-citation" title="Candidate relation source chunk citation">Chunk: ${escapeHtml(chunkId)}</div>` : "";
        return `
            <tr>
                <td><strong>${escapeHtml(r.from_concept)}</strong></td>
                <td><span class="relation-type-tag">${escapeHtml(relType)}</span></td>
                <td><strong>${escapeHtml(r.to_concept)}</strong></td>
                <td><span class="confidence-score">${conf}</span></td>
                <td>
            <div><em>${escapeHtml(evidence)}</em></div>
            <div class="relation-validation-state">${escapeHtml(r.validation_status || (r.is_candidate ? "candidate, unverified" : "validation state unavailable"))}</div>
                    ${citation}
                </td>
            </tr>
        `;
    }).join("");
}

function cleanSpacingDisplay(text) {
    if (!text) return "";
    return String(text)
        .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
        .replace(/([A-Z]{2,})([A-Z][a-z])/g, "$1 $2")
        .replace(/([a-zA-Z])(\d)/g, "$1 $2")
        .replace(/(\d)([a-zA-Z])/g, "$1 $2")
        .replace(/([a-zA-Z0-9])([:;])([a-zA-Z0-9])/g, "$1$2 $3")
        .replace(/\s+/g, " ")
        .trim();
}

function renderDocumentStructure(sections) {
    const container = document.getElementById("structure-container");
    const countBadge = document.getElementById("structure-count");
    container.innerHTML = "";

    if (!sections || sections.length === 0) {
        container.innerHTML = `<div class="muted">No structural hierarchy generated yet.</div>`;
        countBadge.textContent = "0 sections";
        return;
    }

    countBadge.textContent = `${sections.length} sections`;
    container.innerHTML = sections.map((sec) => {
        const cleanTitle = cleanSpacingDisplay(sec.title || "Untitled Section");
        const cleanSummary = cleanSpacingDisplay(sec.summary || "");
        const points = (sec.key_points || []).map(p => `<li>${escapeHtml(cleanSpacingDisplay(p))}</li>`).join("");
        const level = Math.max(1, parseInt(sec.level || 1, 10));
        const indentPx = (level - 1) * 20;
        return `
            <div class="structure-item level-${level}" style="margin-left: ${indentPx}px;">
                <div class="structure-item-header">
                    <span class="section-level-badge">L${level}</span>
                    <span class="structure-item-title">${escapeHtml(cleanTitle)}</span>
                </div>
                ${cleanSummary ? `<div class="structure-item-summary">${escapeHtml(cleanSummary)}</div>` : ""}
                ${points ? `<ul class="structure-points-list">${points}</ul>` : ""}
            </div>
        `;
    }).join("");
}

async function askDocument() {
    const input = document.getElementById("rag-query-input");
    const query = input.value.trim();
    if (!query || !selectedDocId) return;

    const btn = document.getElementById("rag-query-btn");
    const statusBox = document.getElementById("rag-status");
    const outputBox = document.getElementById("rag-output-box");
    const answerText = document.getElementById("rag-answer-text");
    const sourcesList = document.getElementById("rag-sources-list");

    btn.disabled = true;
    btn.textContent = "Searching...";
    statusBox.textContent = "Retrieving relevant context and synthesizing grounded response...";
    statusBox.className = "status-box info";

    try {
        const res = await fetch(`${API_BASE}/query/${encodeURIComponent(selectedDocId)}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, top_k: 4 }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Query execution failed");

        outputBox.style.display = "flex";
        answerText.textContent = data.response || "No response returned.";
        sourcesList.innerHTML = "";

        const results = data.results || [];
        if (results.length === 0) {
            sourcesList.innerHTML = `<div class="muted">No chunk citations returned for this query.</div>`;
        } else {
            results.forEach((r, idx) => {
                const card = document.createElement("div");
                card.className = "source-chunk-card";
                const scoreDisplay = (r.score !== undefined && r.score !== null) ? Number(r.score).toFixed(3) : "Verified";
                const sourceName = escapeHtml(r.document_name || selectedDoc?.name || selectedDoc?.filename || "Source document");
                const spans = Array.isArray(r.source_intervals) ? r.source_intervals : [];
                const citation = spans.length ? spans.map((span) => {
                    const page = span.page_number ? `Page ${escapeHtml(String(span.page_number))}` : "Page unavailable";
                    const offsets = Number.isInteger(span.page_char_start) && Number.isInteger(span.page_char_end)
                        ? ` · page characters ${span.page_char_start}–${span.page_char_end}`
                        : (Number.isInteger(span.source_start) && Number.isInteger(span.source_end)
                            ? ` · extracted-text characters ${span.source_start}–${span.source_end}` : " · source interval unavailable");
                    return `${page}${offsets}`;
                }).join("; ") : "Page and source interval unavailable";
                card.innerHTML = `
                    <div class="source-chunk-header">
                        <span>[Evidence ${idx + 1}] ${escapeHtml(r.section_title || "Section")} (Chunk ${escapeHtml((r.chunk_id || "").slice(0, 8))})</span>
                        <span class="badge" style="background: rgba(56, 189, 248, 0.12); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3);">Relevance: ${scoreDisplay}</span>
                    </div>
                    <div class="source-citation">${sourceName} · ${citation}</div>
                    <div class="source-chunk-snippet">"${escapeHtml((r.content || "").slice(0, 320))}..."</div>
                `;
                sourcesList.appendChild(card);
            });
        }

        const modeBadge = document.getElementById("rag-mode-badge");
        if (data.fallback_used) {
            statusBox.innerHTML = `Grounded response synthesized via deterministic fallback <span class="badge badge-warning" style="background:#f59e0b;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;margin-left:8px;">Fallback Analysis</span>`;
            statusBox.className = "status-box warning";
            if (modeBadge) {
                modeBadge.textContent = "Fallback Analysis";
                modeBadge.style.background = "#f59e0b";
                modeBadge.style.color = "#fff";
            }
        } else {
            statusBox.textContent = "Grounded response synthesized successfully.";
            statusBox.className = "status-box success";
            if (modeBadge) {
                modeBadge.textContent = "RAG Response";
                modeBadge.style.background = "";
                modeBadge.style.color = "";
            }
        }
    } catch (err) {
        statusBox.textContent = `Query error: ${formatFetchError(err, "/query")}`;
        statusBox.className = "status-box error";
    } finally {
        btn.disabled = false;
        btn.textContent = "Ask";
    }
}

function copyRagAnswer() {
    const answerText = document.getElementById("rag-answer-text");
    const copyBtn = document.getElementById("copy-rag-btn");
    if (!answerText || !answerText.textContent) return;
    navigator.clipboard.writeText(answerText.textContent).then(() => {
        if (copyBtn) {
            const original = copyBtn.textContent;
            copyBtn.textContent = "Copied!";
            copyBtn.style.color = "#10b981";
            setTimeout(() => {
                copyBtn.textContent = original;
                copyBtn.style.color = "";
            }, 2000);
        }
    }).catch(err => {
        console.error("Clipboard copy failed", err);
    });
}

function updateOverviewMetric(elementId, value) {
    const el = document.getElementById(elementId);
    if (el) {
        if (value === null || value === undefined || value === "") {
            el.textContent = "--";
        } else {
            el.textContent = typeof value === "number" ? Number(value).toLocaleString() : value;
        }
    }
}

function initPipelineBoard() {
    const container = document.getElementById("pipeline-stages");
    container.innerHTML = "";
    PIPELINE_STAGES.forEach((stage) => {
        const row = document.createElement("div");
        row.className = "stage-row";
        row.dataset.stageKey = stage.key;
        row.innerHTML = `
            <span class="stage-name">${stage.label}</span>
            <span class="stage-status-pill" id="pill-${stage.key}">Pending</span>
            <div class="stage-progress-bar-wrap">
                <div class="stage-progress-bar" id="bar-${stage.key}"></div>
            </div>
            <span class="stage-percent" id="pct-${stage.key}">0%</span>
        `;
        container.appendChild(row);
    });
}

function resetPipelineBoard() {
    PIPELINE_STAGES.forEach((stage) => {
        setStageState(stage.key, "pending", 0);
    });
    document.getElementById("pipeline-summary").textContent = "Waiting for execution";
}

/**
 * Format a stage duration (in seconds) with appropriate precision.
 * - null/undefined → "--" (not zero)
 * - < 1 second      → "Xms"  (e.g. "3ms", "125ms")
 * - >= 1 second     → "X.XXs" (e.g. "1.25s")
 */
function formatStageDuration(durSec) {
    if (durSec === null || durSec === undefined) return "--";
    const ms = Math.round(Number(durSec) * 1000);
    if (ms < 1000) return `${ms}ms`;
    return `${Number(durSec).toFixed(2)}s`;
}

/**
 * Resolve semantically meaningful input/output labels and values for each
 * pipeline stage. Returns { inLabel, inVal, outLabel, outVal }.
 * Values are taken from the stage object without fabrication.
 */
function resolveStageMetrics(stageName, s) {
    const qm = s.quality_metrics || {};
    switch (stageName) {
        case "ingestion":
            return {
                inLabel: "File bytes",
                inVal: s.input_size ?? null,
                outLabel: "Characters extracted",
                outVal: s.output_size ?? qm.char_count ?? null,
            };
        case "chunking":
            return {
                inLabel: "Characters",
                inVal: s.input_size ?? null,
                outLabel: "Chunks produced",
                outVal: s.chunk_count ?? s.output_size ?? qm.chunk_count ?? null,
            };
        case "embedding":
            return {
                inLabel: "Chunks embedded",
                inVal: s.input_size ?? qm.chunk_count ?? null,
                outLabel: "Vectors (dims)",
                outVal: s.output_size != null
                    ? `${s.output_size}${qm.embedding_dim ? " × " + qm.embedding_dim : ""}`
                    : null,
            };
        case "mapping":
            return {
                inLabel: "Chunks analysed",
                inVal: s.input_size ?? null,
                outLabel: "Relations extracted",
                outVal: s.output_size ?? qm.relation_count ?? null,
            };
        case "graph_building":
            return {
                inLabel: "Relations mapped",
                inVal: s.input_size ?? null,
                outLabel: "Nodes / Edges",
                outVal: (s.node_count != null && s.edge_count != null)
                    ? `${s.node_count} / ${s.edge_count}`
                    : (s.output_size ?? null),
            };
        case "evaluation":
            return {
                inLabel: "Candidate relations evaluated",
                inVal: s.input_size ?? qm.candidates_evaluated ?? null,
                outLabel: "Risk flags detected",
                outVal: s.output_size ?? qm.risk_flags ?? null,
            };
        case "hierarchy":
            return {
                inLabel: "Graph elements processed",
                inVal: s.input_size ?? (
                    (qm.node_count != null && qm.edge_count != null)
                        ? qm.node_count + qm.edge_count
                        : null
                ),
                outLabel: "Hierarchy levels generated",
                outVal: s.output_size ?? qm.hierarchy_levels ?? qm.depth ?? null,
            };
        default:
            return {
                inLabel: "Input",
                inVal: s.input_size ?? null,
                outLabel: "Output",
                outVal: s.output_size ?? null,
            };
    }
}

function renderPipelineStatus(payload) {
    if (!payload || !payload.stages) return;
    const stages = payload.stages;

    // Populate timeline table in Technical Details
    const timelineBody = document.getElementById("timeline-table-body");
    timelineBody.innerHTML = "";

    const stageEntries = Object.entries(stages);
    if (stageEntries.length === 0) {
        timelineBody.innerHTML = `<tr><td colspan="5" class="muted text-center">No stage telemetry recorded</td></tr>`;
    } else {
        timelineBody.innerHTML = stageEntries.map(([name, s]) => {
            const durVal = s.duration_sec ?? s.duration_seconds;
            const dur = formatStageDuration(durVal);
            const { inLabel, inVal, outLabel, outVal } = resolveStageMetrics(name, s);
            const statusClass = s.status === "completed" ? "completed" : (s.status === "fallback" ? "fallback" : "pending");
            const displayName = PIPELINE_STAGES.find(ps => ps.key === name)?.label || name;
            return `
                <tr>
                    <td><strong>${escapeHtml(displayName)}</strong></td>
                    <td><span class="stage-status-pill ${statusClass}">${escapeHtml(s.status || "pending")}</span></td>
                    <td><span class="confidence-score">${escapeHtml(dur)}</span></td>
                    <td title="${escapeHtml(inLabel)}">${escapeHtml(inLabel)}: ${inVal !== null && inVal !== undefined ? inVal : "--"}</td>
                    <td title="${escapeHtml(outLabel)}">${escapeHtml(outLabel)}: ${outVal !== null && outVal !== undefined ? outVal : "--"}</td>
                </tr>
            `;
        }).join("");
    }

    // Update Telemetry grid
    if (payload.active_model) document.getElementById("telem-llm-model").textContent = payload.active_model;
    if (payload.analysis_mode) document.getElementById("telem-analysis-mode").textContent = payload.analysis_mode;

    // Update chunk count in overview if available
    if (stages.chunking && stages.chunking.chunk_count) {
        updateOverviewMetric("stat-chunks", stages.chunking.chunk_count);
    }

    // Update visual progress bars
    for (const [stageName, stageData] of Object.entries(stages)) {
        const matchingStage = PIPELINE_STAGES.find((s) => s.key === stageName || s.key.startsWith(stageName));
        if (matchingStage) {
            const pct = stageData.status === "completed" || stageData.status === "fallback" ? 100 : (stageData.progress || 50);
            setStageState(matchingStage.key, stageData.status, pct);
        }
    }
}

function setStageState(stageKey, stageStatus, progress) {
    const pill = document.getElementById(`pill-${stageKey}`);
    const bar = document.getElementById(`bar-${stageKey}`);
    const pct = document.getElementById(`pct-${stageKey}`);

    if (pill) {
        pill.textContent = stageStatus;
        pill.className = `stage-status-pill ${stageStatus}`;
    }
    if (bar) {
        bar.style.width = `${Math.min(100, Math.max(0, progress))}%`;
    }
    if (pct) {
        pct.textContent = `${Math.round(progress)}%`;
    }
}

function formatLogEntry(raw) {
    if (typeof raw === "string") {
        try {
            const item = JSON.parse(raw);
            return formatLogObject(item);
        } catch {
            return raw;
        }
    }
    return formatLogObject(raw);
}

function formatLogObject(obj) {
    if (!obj || typeof obj !== "object") return String(obj);
    let time = "";
    if (obj.timestamp) {
        time = obj.timestamp.includes(" ") ? obj.timestamp.split(" ")[1] : (obj.timestamp.split("T")[1]?.slice(0, 8) || obj.timestamp);
    }
    const level = obj.level ? `[${obj.level}]` : "[INFO]";
    const stage = obj.stage && obj.stage !== "system" ? `[${obj.stage.toUpperCase()}]` : "";
    const msg = obj.message || JSON.stringify(obj);
    return `${time ? time + " " : ""}${level} ${stage ? stage + " " : ""}${msg}`.trim();
}

async function loadInitialLogs() {
    try {
        const response = await fetch(`${API_BASE}/logs/recent?lines=60`);
        if (response.ok) {
            const payload = await response.json();
            const consoleBox = document.getElementById("logs-console");
            const rawLines = payload.lines || payload.logs;
            if (rawLines && Array.isArray(rawLines) && consoleBox) {
                consoleBox.textContent = rawLines.map(formatLogEntry).join("\n") + "\n";
                consoleBox.scrollTop = consoleBox.scrollHeight;
            }
        }
    } catch {
        // silent
    }
}

function connectLogStream() {
    if (logSocket) {
        logSocket.close();
        logSocket = null;
    }
    stopLogPolling();
    loadInitialLogs();

    const statusBadge = document.getElementById("log-stream-status");

    try {
        logSocket = new WebSocket(`${WS_BASE}/logs/stream`);
        logSocket.onopen = () => {
            if (statusBadge) {
                statusBadge.textContent = "● Live Stream";
                statusBadge.style.color = "#10b981";
            }
        };
        logSocket.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                appendLogLine(payload);
            } catch {
                appendRawLog(event.data);
            }
        };
        logSocket.onerror = () => {
            startLogPolling();
        };
        logSocket.onclose = () => {
            startLogPolling();
        };
    } catch {
        startLogPolling();
    }
}

function startLogPolling() {
    if (logPollTimer) return;
    const statusBadge = document.getElementById("log-stream-status");
    if (statusBadge) {
        statusBadge.textContent = "● Live Polling";
        statusBadge.style.color = "#38bdf8";
    }
    logPollTimer = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/logs/recent?lines=80`);
            if (response.ok) {
                const payload = await response.json();
                const consoleBox = document.getElementById("logs-console");
                const rawLines = payload.lines || payload.logs;
                if (rawLines && Array.isArray(rawLines) && consoleBox) {
                    consoleBox.textContent = rawLines.map(formatLogEntry).join("\n") + "\n";
                    consoleBox.scrollTop = consoleBox.scrollHeight;
                }
            }
        } catch {
            // silent
        }
    }, 2000);
}

function stopLogPolling() {
    if (logPollTimer) {
        clearInterval(logPollTimer);
        logPollTimer = null;
    }
}

function appendLogLine(payload) {
    const consoleBox = document.getElementById("logs-console");
    if (!consoleBox) return;
    const line = formatLogEntry(payload);
    consoleBox.textContent += line + "\n";
    consoleBox.scrollTop = consoleBox.scrollHeight;
    trimLogs(consoleBox);
}

function appendRawLog(raw) {
    const consoleBox = document.getElementById("logs-console");
    if (!consoleBox) return;
    const line = formatLogEntry(raw);
    consoleBox.textContent += line + "\n";
    consoleBox.scrollTop = consoleBox.scrollHeight;
    trimLogs(consoleBox);
}

function trimLogs(consoleBox) {
    const lines = consoleBox.textContent.split("\n");
    if (lines.length > 400) {
        consoleBox.textContent = lines.slice(-300).join("\n");
    }
}

function initGraph() {
    graph = cytoscape({
        container: document.getElementById("graph-container"),
        elements: [],
        style: [
            {
                selector: "node",
                style: {
                    "background-color": function (ele) {
                        const type = ele.data("type");
                        if (type === "topic") return "#0ea5e9";
                        if (type === "subtopic") return "#0284c7";
                        if (type === "document") return "#f97316";
                        return "#10b981";
                    },
                    shape: "round-rectangle",
                    label: function (ele) {
                        return graphDisplayLabel(ele.data());
                    },
                    color: "#f8fafc",
                    "text-valign": "center",
                    "text-halign": "center",
                    "text-wrap": "wrap",
                    "text-max-width": 140,
                    "font-size": "11px",
                    "font-weight": "600",
                    width: 145,
                    height: 44,
                    "border-width": 1.5,
                    "border-color": "#334155",
                    padding: "8px",
                },
            },
            {
                selector: "edge",
                style: {
                    width: function (ele) {
                        return ele.data("relation_type") === "hierarchy" ? 2.0 : 1.2;
                    },
                    "line-color": function (ele) {
                        return ele.data("relation_type") === "hierarchy" ? "#475569" : "#38bdf8";
                    },
                    "target-arrow-color": function (ele) {
                        return ele.data("relation_type") === "hierarchy" ? "#475569" : "#38bdf8";
                    },
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    opacity: 0.6,
                    "line-style": "solid",
                    "font-size": 8,
                    "text-rotation": "autorotate",
                    "text-background-color": "#0b1220",
                    "text-background-opacity": 0.85,
                    "text-background-padding": 2,
                },
            },
            { selector: 'edge[is_candidate = true]', style: { "line-style": "dashed", "line-dash-pattern": [6, 4], "line-color": "#f59e0b", "target-arrow-color": "#f59e0b" } },
            { selector: 'edge[validated = true]', style: { "line-style": "solid", "line-color": "#34d399", "target-arrow-color": "#34d399" } },
            { selector: ".filter-hidden", style: { display: "none" } },
            {
                selector: 'edge[relation_type = "hierarchy"]',
                style: {
                    "target-arrow-shape": "none",
                },
            },
            {
                selector: ".highlighted",
                style: {
                    "border-width": 3,
                    "border-color": "#38bdf8",
                    "background-color": "#0284c7",
                    "box-shadow": "0 0 12px #38bdf8",
                },
            },
            {
                selector: ".dimmed",
                style: {
                    opacity: 0.15,
                },
            },
            {
                selector: ".neighbor",
                style: {
                    "border-width": 2.5,
                    "border-color": "#38bdf8",
                    opacity: 1.0,
                },
            },
            {
                selector: ".selected-focus",
                style: {
                    "border-width": 3.5,
                    "border-color": "#38bdf8",
                    "background-color": "#0284c7",
                    opacity: 1.0,
                    label: function (ele) { return ele.isEdge() ? (ele.data("relation_type") || "relationship") : graphDisplayLabel(ele.data()); },
                },
            },
        ],
        layout: {
            name: "concentric",
            padding: 40,
            fit: true,
        },
        autoungrabify: true,
        userZoomingEnabled: true,
        userPanningEnabled: true,
    });

    graph.on("mouseover", "node", function (e) {
        const data = e.target.data();
        const tooltip = document.getElementById("graph-tooltip");
        const definition = data.definition || data.description || "No source description is available.";
        tooltip.innerHTML = `
            <strong>${escapeHtml(data.label || data.name || data.id)}</strong><br>
            <span class="badge">${escapeHtml(data.type || "concept")}</span>
            <div style="font-size: 0.78rem; margin-top: 6px; color: #cbd5e1">${escapeHtml(definition)}</div>
        `;
        tooltip.style.display = "block";
    });

    graph.on("mouseover", "edge", function (e) {
        const data = e.target.data();
        const tooltip = document.getElementById("graph-tooltip");
        tooltip.innerHTML = `
            <strong>${escapeHtml(data.relation_type || "connection")}${data.is_candidate ? " · candidate" : ""}</strong><br>
            <div style="font-size: 0.78rem; margin-top: 4px; color: #cbd5e1">${escapeHtml(data.description || "")}</div>
        `;
        tooltip.style.display = "block";
    });

    graph.on("mousemove", function (e) {
        const tooltip = document.getElementById("graph-tooltip");
        if (tooltip.style.display === "block") {
            tooltip.style.left = (e.originalEvent.pageX + 12) + "px";
            tooltip.style.top = (e.originalEvent.pageY + 12) + "px";
        }
    });

    graph.on("mouseout", "node, edge", function () {
        document.getElementById("graph-tooltip").style.display = "none";
    });

    // Inspector click and 1-hop neighborhood highlight
    graph.on("tap", "node", function (e) {
        focusGraphNode(e.target);
        inspectNode(e.target.data());
    });
    graph.on("tap", "edge", function (e) {
        focusGraphEdge(e.target);
        inspectRelation(e.target.data());
    });

    // Click canvas background to reset focus
    graph.on("tap", function (e) {
        if (e.target === graph) {
            resetGraphFocus();
        }
    });
}

function focusGraphNode(node) {
    if (!graph || !node) return;
    graph.elements().removeClass("dimmed neighbor selected-focus highlighted");
    const neighborhood = node.closedNeighborhood();
    graph.elements().not(neighborhood).addClass("dimmed");
    neighborhood.nodes().not(node).addClass("neighbor");
    node.addClass("selected-focus");
}

function focusGraphEdge(edge) {
    if (!graph || !edge) return;
    graph.elements().removeClass("dimmed neighbor selected-focus highlighted");
    const neighborhood = edge.connectedNodes().union(edge);
    graph.elements().not(neighborhood).addClass("dimmed");
    neighborhood.nodes().addClass("neighbor");
    edge.addClass("selected-focus");
}

function resetGraphFocus() {
    if (!graph) return;
    graph.elements().removeClass("dimmed neighbor selected-focus highlighted");
}

function focusNodeByName(conceptName) {
    if (!graph || !conceptName) return;
    const target = graph.nodes().filter(n => {
        const d = n.data();
        return (d.label || d.name || d.id || "").toLowerCase() === conceptName.toLowerCase();
    });
    if (target.length > 0) {
        focusGraphNode(target.first());
        graph.animate({
            center: { eles: target.first() },
            zoom: Math.max(graph.zoom(), 1.0),
            duration: 400
        });
        inspectNode(target.first().data());
    }
}

function applyGraphLayout(layoutName) {
    if (!graph) return;
    const validLayouts = ["concentric", "breadthfirst", "cose"];
    const name = validLayouts.includes(layoutName) ? layoutName : "concentric";
    const layoutOptions = {
        name: name,
        animate: true,
        animationDuration: 500,
        padding: 40,
        fit: true,
    };
    if (name === "breadthfirst") {
        layoutOptions.directed = true;
        layoutOptions.spacingFactor = 1.3;
    } else if (name === "concentric") {
        layoutOptions.concentric = function(node) {
            return node.data("type") === "document" ? 3 : (node.data("type") === "topic" ? 2 : 1);
        };
        layoutOptions.levelWidth = function() { return 1; };
    } else if (name === "cose") {
        layoutOptions.nodeRepulsion = () => 8000;
        layoutOptions.idealEdgeLength = () => 110;
        layoutOptions.componentSpacing = 90;
        layoutOptions.numIter = 500;
    }
    graph.layout(layoutOptions).run();
}

function inspectNode(data) {
    const title = document.getElementById("inspector-title");
    const badge = document.getElementById("inspector-badge");
    const body = document.getElementById("inspector-body");

    const nodeName = data.label || data.name || data.id;
    title.textContent = nodeName;
    badge.textContent = `Type: ${data.type || "concept"}`;

    // Match selected node to currentKnowledgeMap.concepts
    let matchedConcept = null;
    if (currentKnowledgeMap && currentKnowledgeMap.concepts) {
        matchedConcept = currentKnowledgeMap.concepts.find(c =>
            (c.name && c.name.toLowerCase() === nodeName.toLowerCase()) ||
            (c.id && c.id.toLowerCase() === (data.id || "").toLowerCase()) ||
            (c.name && c.name.toLowerCase() === (data.id || "").toLowerCase())
        );
    }

    const definition = (matchedConcept && (matchedConcept.definition || matchedConcept.context)) || data.definition || data.description || "No source description is available.";
    const occurrences = matchedConcept?.frequency !== undefined ? `${matchedConcept.frequency} occurrences` : (data.frequency !== undefined ? `${data.frequency} occurrences` : "Occurrence count unavailable");
    const supportingSections = (matchedConcept && matchedConcept.supporting_sections) || data.supporting_sections || [];
    const chunkIds = (matchedConcept && matchedConcept.chunk_ids) || data.chunk_ids || [];

    // Find connected relations in currentKnowledgeMap
    let relationsHtml = "";
    if (currentKnowledgeMap && currentKnowledgeMap.relations) {
        const connected = currentKnowledgeMap.relations.filter(
            r => (r.from_concept || "").toLowerCase() === nodeName.toLowerCase() || (r.to_concept || "").toLowerCase() === nodeName.toLowerCase()
        );
        if (connected.length > 0) {
            relationsHtml = `
                <div class="inspector-relations-list">
                    <strong>Connected Candidate Relations (${connected.length}):</strong>
                    ${connected.slice(0, 6).map(r => {
                        const targetConcept = (r.from_concept || "").toLowerCase() === nodeName.toLowerCase() ? r.to_concept : r.from_concept;
                        return `
                        <div class="relation-item-pill" onclick="focusNodeByName('${escapeHtml(targetConcept)}')" title="Click to focus ${escapeHtml(targetConcept)} in graph">
                            <span class="relation-type-tag">${escapeHtml(r.relation_type)}</span>
                            <span>${escapeHtml(r.from_concept)} ➔ ${escapeHtml(r.to_concept)}</span>
                            <span class="confidence-score">${Number(r.confidence || 0).toFixed(2)}</span>
                        </div>
                        `;
                    }).join("")}
                </div>
            `;
        }
    }

    const sourceEvidence = matchedConcept?.source_evidence || [];
    const evidenceHtml = sourceEvidence.length ? `
        <div class="inspector-citations">
            <strong>Source passages</strong>
            ${sourceEvidence.map(item => {
                const pages = (item.page_references || []).filter(Number.isInteger);
                const links = pages.length ? pages.map(page => `<a class="source-page-link" href="${API_BASE}/upload/${encodeURIComponent(selectedDocId)}/source#page=${page}" target="_blank" rel="noopener">Open PDF · page ${page}</a>`).join(" ") : `<span class="muted">Page location unavailable</span>`;
                return `<blockquote><p>${escapeHtml(item.excerpt || "")}</p><small>${escapeHtml(item.section_title || "Extracted passage")} · chunk ${escapeHtml(item.chunk_id || "")}</small><div>${links}</div></blockquote>`;
            }).join("")}
        </div>` : `<p class="muted">No chunk-level source passage is available for this node.</p>`;

    // Source Grounding Citations
    let citationsHtml = "";
    if (supportingSections.length > 0 || chunkIds.length > 0) {
        citationsHtml = `
            <div class="inspector-citations" style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 6px; border: 1px solid var(--border);">
                <div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; color: var(--text-dim); margin-bottom: 6px;">Source Grounding & Citations</div>
                ${supportingSections.length > 0 ? `<div style="margin-bottom: 4px;"><strong>Sections:</strong> ${supportingSections.map(s => `<span class="badge" style="margin-right:4px;">${escapeHtml(s)}</span>`).join("")}</div>` : ""}
                ${chunkIds.length > 0 ? `<div><strong>Source Chunks:</strong> ${chunkIds.map(cid => `<code class="chunk-citation" style="margin-right:4px;">${escapeHtml(cid)}</code>`).join("")}</div>` : ""}
            </div>
        `;
    }

    body.innerHTML = `
        <p><strong>Definition & Context:</strong> ${escapeHtml(definition)}</p>
        <p><strong>Frequency / Occurrences:</strong> ${escapeHtml(occurrences)}</p>
        ${citationsHtml}
        ${evidenceHtml}
        ${relationsHtml}
    `;
}

function inspectRelation(data) {
    const relationId = String(data.id || "").replace(/^rel_/, "");
    const relation = (currentKnowledgeMap?.relations || []).find(item => item.id === relationId);
    const title = document.getElementById("inspector-title");
    const badge = document.getElementById("inspector-badge");
    const body = document.getElementById("inspector-body");
    title.textContent = relation ? `${relation.from_concept} → ${relation.to_concept}` : "Document structure connection";
    badge.textContent = relation ? `${relation.relation_type} · candidate, unverified` : "Structural grouping";
    if (!relation) {
        body.innerHTML = `<p>This edge represents document or topic hierarchy. It is not a semantic claim.</p>`;
        return;
    }
    const evidence = relation.source_evidence || [];
    const evidenceCards = evidence.map(item => {
        const pages = (item.page_references || []).filter(Number.isInteger);
        const links = pages.map(page => `<a class="source-page-link" href="${API_BASE}/upload/${encodeURIComponent(selectedDocId)}/source#page=${page}" target="_blank" rel="noopener">Open PDF · page ${page}</a>`).join(" ");
        return `<blockquote><p>${escapeHtml(item.excerpt || "")}</p><small>${escapeHtml(item.section_title || "Extracted passage")} · chunk ${escapeHtml(item.chunk_id || "")}</small><div>${links || "Page location unavailable"}</div></blockquote>`;
    }).join("");
    body.innerHTML = `<p><strong>Candidate relationship:</strong> ${escapeHtml(relation.from_concept)} ${escapeHtml(relation.relation_type)} ${escapeHtml(relation.to_concept)}</p><p><strong>Status:</strong> Not independently validated. Confidence ${Number(relation.confidence).toFixed(2)} is a system score, not a human-verified probability.</p><p><strong>Evidence passage:</strong> ${escapeHtml(relation.evidence_text || "No supporting sentence was retained.")}</p>${evidenceCards || `<p class="muted">No source chunk is attached to this relationship.</p>`}`;
}


async function loadGraph(documentId) {
    if (!isDocumentAnalyzed(selectedDoc)) {
        resetGraphPanels("Graph will appear after analysis completes.");
        return;
    }
    try {
        const response = await fetch(`${API_BASE}/graph/${documentId}`);
        if (!response.ok) {
            resetGraphPanels("No graph available.");
            return;
        }
        const payload = await response.json();
        const elements = simplifyGraphElements(toGraphElements(payload));
        graph.elements().remove();
        graph.add(elements);
        applyGraphFilters();
        const layoutSelect = document.getElementById("graph-layout");
        const activeLayout = layoutSelect ? layoutSelect.value || "concentric" : "concentric";
        applyGraphLayout(activeLayout);
        graph.autoungrabify(true);

        const nodeCount = graph.nodes().length;
        const edgeCount = graph.edges().length;
        document.getElementById("graph-info").textContent = `${nodeCount} nodes / ${edgeCount} edges`;

        updateOverviewMetric("stat-nodes", nodeCount);
        updateOverviewMetric("stat-edges", edgeCount);
    } catch (error) {
        document.getElementById("graph-info").textContent = "Graph load failed.";
    }
}

function resetGraphPanels(message) {
    document.getElementById("graph-info").textContent = message;
    if (graph) graph.elements().remove();
}

function applyGraphFilters() {
    if (!graph) return;
    const query = (document.getElementById("graph-search")?.value || "").trim().toLowerCase();
    const nodeType = document.getElementById("graph-node-filter")?.value || "all";
    const edgeType = document.getElementById("graph-edge-filter")?.value || "all";
    graph.nodes().forEach(node => {
        const data = node.data();
        const matches = (nodeType === "all" || data.type === nodeType) && (!query || graphDisplayLabel(data).toLowerCase().includes(query));
        node.toggleClass("filter-hidden", !matches);
    });
    graph.edges().forEach(edge => {
        const relationType = edge.data("relation_type");
        const isHierarchy = relationType === "hierarchy";
        const matches = edgeType === "all" || (edgeType === "hierarchy" && isHierarchy) || (edgeType === "candidate" && !isHierarchy);
        const hiddenEndpoint = edge.source().hasClass("filter-hidden") || edge.target().hasClass("filter-hidden");
        edge.toggleClass("filter-hidden", !matches || hiddenEndpoint);
    });
}

function simplifyGraphElements(elementsArr) {
    return elementsArr;
}

function graphDisplayLabel(data) {
    return data.label || data.name || data.id || "";
}

function normalizeGraphNode(node, index) {
    const d = node.data || {};
    return {
        data: {
            id: d.id || `node-${index}`,
            label: d.label || d.name || d.id || `Node ${index}`,
            type: d.type || "concept",
            definition: d.definition || d.description || "",
            frequency: d.frequency || 1,
            centrality: d.centrality || 0,
        },
    };
}

function normalizeGraphEdge(edge, index) {
    const d = edge.data || {};
    return {
        data: {
            id: d.id || `edge-${index}`,
            source: d.source,
            target: d.target,
            relation_type: d.relation_type || d.type || "semantic",
            description: d.description || "",
            confidence: d.confidence || 0.6,
        },
    };
}

function toGraphElements(payload) {
    const elements = [];
    const rawNodes = payload?.elements?.nodes || [];
    const rawEdges = payload?.elements?.edges || [];

    rawNodes.forEach((n, idx) => elements.push(normalizeGraphNode(n, idx)));
    rawEdges.forEach((e, idx) => elements.push(normalizeGraphEdge(e, idx)));

    return elements;
}

async function refreshSystemStatus() {
    try {
        const [sysRes, healthRes] = await Promise.all([
            fetch(`${API_BASE}/system-status`),
            fetch(`${API_BASE}/system/health-check${selectedDocId ? `?document_id=${encodeURIComponent(selectedDocId)}` : ""}`),
        ]);

        if (healthRes.ok) {
            const hData = await healthRes.json();
            const healthPill = document.getElementById("health-pill");
            const dot = document.getElementById("system-dot");
            const statusStr = String(hData.status || "healthy").toLowerCase();
            healthPill.textContent = `System: ${statusStr}`;

            if (statusStr === "healthy") {
                dot.className = "status-indicator-dot healthy";
            } else if (statusStr === "degraded" || statusStr === "warning") {
                dot.className = "status-indicator-dot degraded";
            } else {
                dot.className = "status-indicator-dot error";
            }

            const llmCheck = (hData.checks && hData.checks.llm_connectivity) || {};
            const llmAvailable = llmCheck.available ?? (llmCheck.status === "healthy");

            const modeText = document.getElementById("status-mode")?.textContent || "";
            if (modeText.includes("Fallback")) {
                updateHonestStatus("Fallback Active", "fallback");
            } else if (llmCheck.status === "unhealthy" || llmAvailable === false) {
                updateHonestStatus("LLM Unavailable", "unavailable");
            } else if (llmCheck.status === "warning") {
                updateHonestStatus("LLM Degraded", "degraded");
            } else {
                updateHonestStatus("LLM Ready", "ready");
            }
        }
    } catch {
        const dot = document.getElementById("system-dot");
        if (dot) dot.className = "status-indicator-dot error";
        document.getElementById("health-pill").textContent = "System: offline";
        updateHonestStatus("LLM Unavailable", "unavailable");
    }
}

function isDocumentAnalyzed(doc) {
    return String(doc?.status || "").toLowerCase() === "analyzed";
}

async function loadDatasetStats() {
    const box = document.getElementById("dataset-stats");
    if (!box) return;
    const jsonBtn = document.getElementById("export-json-btn");
    const csvBtn = document.getElementById("export-csv-btn");

    if (jsonBtn) {
        jsonBtn.href = `${API_BASE}/dataset/relations/export?format=json${selectedDocId ? "&document_id=" + encodeURIComponent(selectedDocId) : ""}`;
    }
    if (csvBtn) {
        csvBtn.href = `${API_BASE}/dataset/relations/export?format=csv${selectedDocId ? "&document_id=" + encodeURIComponent(selectedDocId) : ""}`;
    }

    try {
        const response = await fetch(`${API_BASE}/dataset/relations/stats`);
        if (!response.ok) return;
        const stats = await response.json();
        if (box) {
            box.innerHTML = `
                <div style="font-size: 0.82rem; line-height: 1.6;">
                    <div><strong>Total Records:</strong> ${Number(stats.total_records || 0).toLocaleString()} candidate relations</div>
                    <div><strong>Unreviewed Pool:</strong> ${Number(stats.unlabeled_records || 0).toLocaleString()} (100% research integrity)</div>
                    <div><strong>Avg LLM Confidence:</strong> ${(Number(stats.average_llm_confidence || 0) * 100).toFixed(1)}%</div>
                    <div><strong>Avg Co-occurrence:</strong> ${(Number(stats.average_cooccurrence || 0) * 100).toFixed(1)}%</div>
                </div>
            `;
        }
    } catch (err) {
        if (box) box.textContent = "Dataset statistics unavailable";
    }
}

function renderDatasetStats(stats) {
    loadDatasetStats();
}

async function loadAnalysisReport() {
    const reportBox = document.getElementById("analysis-report");
    if (!selectedDocId) {
        reportBox.textContent = "Select a document to view report details.";
        return;
    }
    try {
        const response = await fetch(
            `${API_BASE}/system/analysis-report/${encodeURIComponent(selectedDocId)}`
        );
        if (!response.ok) {
            reportBox.textContent = "Analysis report not available yet.";
            return;
        }
        const payload = await response.json();
        renderAnalysisReport(payload);
    } catch (error) {
        reportBox.textContent = "Failed to load analysis report.";
    }
}

function renderAnalysisReport(report) {
    const reportBox = document.getElementById("analysis-report");
    if (!report) return;

    reportBox.textContent = JSON.stringify(report, null, 2);

    const metrics = report.metrics || {};
    const summary = report.summary || {};
    const artifacts = report.artifacts || {};

    const words = metrics.words ?? artifacts.document?.word_count ?? selectedDoc?.word_count;
    if (words !== undefined && words !== null) updateOverviewMetric("stat-words", words);

    const pages = metrics.pages ?? artifacts.document?.page_count ?? selectedDoc?.page_count ?? selectedDoc?.pages;
    if (pages !== undefined && pages !== null) updateOverviewMetric("stat-pages", pages);

    const sections = metrics.sections ?? artifacts.summary?.section_count;
    if (sections !== undefined && sections !== null) updateOverviewMetric("stat-sections", sections);

    const chunks = metrics.chunks ?? artifacts.chunks?.count;
    if (chunks !== undefined && chunks !== null) updateOverviewMetric("stat-chunks", chunks);

    const concepts = metrics.concepts ?? summary.concept_count;
    if (concepts !== undefined && concepts !== null) updateOverviewMetric("stat-concepts", concepts);

    const relations = metrics.candidate_relations ?? metrics.relations ?? summary.relation_count;
    if (relations !== undefined && relations !== null) updateOverviewMetric("stat-relations", relations);

    const nodes = metrics.graph_nodes ?? summary.graph_nodes;
    if (nodes !== undefined && nodes !== null) updateOverviewMetric("stat-nodes", nodes);

    const edges = metrics.graph_edges ?? summary.graph_edges;
    if (edges !== undefined && edges !== null) updateOverviewMetric("stat-edges", edges);

    renderDatasetStats({
        total_records: Number(artifacts.dataset_records || 0),
        exportFolder: artifacts.dataset_export ? "Exported to disk" : "Pending export",
        latestExport: artifacts.dataset_export || "None",
    });
}

function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function setStatus(elementId, message, type = "") {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.textContent = message;
    el.className = `status-box ${type}`;
}

function updateHonestStatus(statusText, variant) {
    const badge = document.getElementById("honest-status-badge");
    const chip = document.getElementById("honest-chip");

    if (badge) badge.textContent = statusText;
    if (chip) {
        chip.className = `status-chip honest-${variant || "ready"}`;
    }
}
