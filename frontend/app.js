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

    document.getElementById("relation-filter").addEventListener("change", (e) => {
        applyRelationFilter(e.target.value);
    });

    document.getElementById("zoom-in").addEventListener("click", () => {
        if (graph) graph.zoom(graph.zoom() * 1.25);
    });
    document.getElementById("zoom-out").addEventListener("click", () => {
        if (graph) graph.zoom(graph.zoom() * 0.8);
    });
    document.getElementById("fit-graph").addEventListener("click", () => {
        if (graph) graph.fit(undefined, 35);
    });

    // Real-time graph concept search
    const searchInput = document.getElementById("graph-search");
    if (searchInput) {
        searchInput.addEventListener("input", (e) => {
            filterGraphSearch(e.target.value);
        });
    }

    // Grounded RAG Query events
    const ragBtn = document.getElementById("rag-query-btn");
    const ragInput = document.getElementById("rag-query-input");
    if (ragBtn && ragInput) {
        ragBtn.addEventListener("click", askDocument);
        ragInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") askDocument();
        });
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

        if (!selectedDocId && documents.length > 0) {
            selectDocument(documents[0]);
        }
        return documents;
    } catch (error) {
        container.innerHTML = `<div class="muted">${escapeHtml(formatFetchError(error, "/upload/list"))}</div>`;
        return [];
    }
}

function selectDocument(doc) {
    selectedDocId = doc.id;
    selectedDoc = doc;
    document.querySelectorAll(".document-item").forEach((item) => item.classList.remove("selected"));
    const matching = document.querySelector(`.document-item[data-doc-id="${selectedDocId}"]`);
    if (matching) matching.classList.add("selected");

    const sizeFormatted = doc.size ? (doc.size / 1024 / 1024).toFixed(2) : "0.00";
    const docPages = doc.page_count ?? doc.pages;
    document.getElementById("doc-info").innerHTML = `
        <strong>${escapeHtml(doc.name)}</strong><br>
        Status: <span class="badge">${escapeHtml(doc.status || 'uploaded')}</span><br>
        Size: ${sizeFormatted} MB • Pages: ${docPages ?? 'N/A'} • Words: ${doc.word_count ? Number(doc.word_count).toLocaleString() : 'N/A'}
    `;

    document.getElementById("status-doc-name").textContent = doc.name;
    document.getElementById("overview-doc-title").textContent = `${doc.name} (${doc.id})`;

    if (doc.word_count) updateOverviewMetric("stat-words", doc.word_count);
    if (docPages) updateOverviewMetric("stat-pages", docPages);

    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = isDocumentAnalyzed(doc) ? "Re-run Full Analysis" : "Start Full Analysis";

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

function resetWorkspaceViews(message) {
    resetGraphPanels(message);
    document.getElementById("overall-summary-box").textContent = message;
    document.getElementById("key-insights-list").innerHTML = `<li class="muted">${escapeHtml(message)}</li>`;
    document.getElementById("doc-type-badge").textContent = "Type: Pending";
    document.getElementById("insights-mode-badge").textContent = "Analysis: Pending";
    document.getElementById("insights-mode-badge").className = "mode-badge";
    document.getElementById("analysis-report").textContent = message;
    document.getElementById("structure-container").innerHTML = `<div class="muted">${escapeHtml(message)}</div>`;
    document.getElementById("structure-count").textContent = "0 sections";

    // Reset 8 metrics
    ["words", "pages", "sections", "chunks", "concepts", "relations", "nodes", "edges"].forEach((m) => {
        const el = document.getElementById(`stat-${m}`);
        if (el) el.textContent = "0";
    });

    renderDatasetStats({
        total_records: 0,
        labeled_records: 0,
        unlabeled_records: 0,
        exportFolder: "Pending analysis",
        latestExport: "No export yet",
    });
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
                                : ["ingestion", "chunking", "embedding", "llm_analysis", "graph_building"];
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
    } catch (err) {
        console.error("Failed to load document summary", err);
    }
}

function renderSummaryAndInsights(summary) {
    const typeBadge = document.getElementById("doc-type-badge");
    const modeBadge = document.getElementById("insights-mode-badge");
    const overallBox = document.getElementById("overall-summary-box");
    const insightsList = document.getElementById("key-insights-list");

    const docType = summary.document_type || "unknown";
    typeBadge.textContent = `Type: ${docType.replace("_", " ")}`;

    const isFallback = (summary.overall_summary || "").includes("Deterministic fallback analysis");
    if (isFallback) {
        modeBadge.textContent = "Mode: Deterministic Fallback";
        modeBadge.className = "mode-badge";
    } else {
        modeBadge.textContent = "Mode: LLM Grounded";
        modeBadge.className = "mode-badge llm-mode";
    }

    overallBox.textContent = summary.overall_summary || "No synthesized summary available.";

    // Generate 3-5 grounded bullet points from sections and summary
    const bullets = [];
    if (summary.overall_summary) {
        const sentences = summary.overall_summary.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 15);
        if (sentences.length > 0) bullets.push(sentences[0]);
        if (sentences.length > 1 && !isFallback) bullets.push(sentences[1]);
    }

    const sections = summary.sections || [];
    for (const sec of sections.slice(0, 4)) {
        if (sec.key_points && sec.key_points.length > 0) {
            bullets.push(`${sec.title}: ${sec.key_points[0]}`);
        } else if (sec.summary) {
            bullets.push(`${sec.title}: ${sec.summary}`);
        }
        if (bullets.length >= 5) break;
    }

    if (bullets.length === 0) {
        insightsList.innerHTML = `<li class="muted">No key structural insights extracted.</li>`;
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
        const definition = c.definition || c.context || "Identified recurring technical terminology.";

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
    const tbody = document.querySelector("#relations-table tbody");
    const countBadge = document.getElementById("relations-count");
    tbody.innerHTML = "";

    if (!relations || relations.length === 0) {
        tbody.innerHTML = `<tr><td colspan="5" class="muted text-center">No candidate relations loaded</td></tr>`;
        countBadge.textContent = "0 relations";
        return;
    }

    countBadge.textContent = `${relations.length} relations`;
    const sorted = [...relations].sort((a, b) => (Number(b.confidence || 0) - Number(a.confidence || 0)));

    tbody.innerHTML = sorted.map((r) => {
        const relType = r.relation_type || "explanation";
        const conf = Number(r.confidence || 0).toFixed(2);
        const evidence = r.evidence_text || r.description || "Inferred from co-occurrence in document context.";
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
                    ${citation}
                </td>
            </tr>
        `;
    }).join("");
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
        const points = (sec.key_points || []).map(p => `<li>${escapeHtml(p)}</li>`).join("");
        const level = Math.max(1, parseInt(sec.level || 1, 10));
        const indentPx = (level - 1) * 20;
        return `
            <div class="structure-item level-${level}" style="margin-left: ${indentPx}px;">
                <div class="structure-item-header">
                    <span class="section-level-badge">L${level}</span>
                    <span class="structure-item-title">${escapeHtml(sec.title || "Untitled Section")}</span>
                </div>
                ${sec.summary ? `<div class="structure-item-summary">${escapeHtml(sec.summary)}</div>` : ""}
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
                const scoreDisplay = r.score ? Number(r.score).toFixed(3) : "N/A";
                card.innerHTML = `
                    <div class="source-chunk-header">
                        <span>[Evidence ${idx + 1}] ${escapeHtml(r.section_title || "Section")} (Chunk ${escapeHtml((r.chunk_id || "").slice(0, 8))})</span>
                        <span>Relevance: ${scoreDisplay}</span>
                    </div>
                    <div class="source-chunk-snippet">"${escapeHtml((r.content || "").slice(0, 260))}..."</div>
                `;
                sourcesList.appendChild(card);
            });
        }

        statusBox.textContent = "Grounded response synthesized successfully.";
        statusBox.className = "status-box success";
    } catch (err) {
        statusBox.textContent = `Query error: ${formatFetchError(err, "/query")}`;
        statusBox.className = "status-box error";
    } finally {
        btn.disabled = false;
        btn.textContent = "Ask";
    }
}

function updateOverviewMetric(elementId, value) {
    const el = document.getElementById(elementId);
    if (el) {
        el.textContent = typeof value === "number" ? Number(value).toLocaleString() : value;
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
            const dur = durVal !== undefined && durVal !== null ? `${Number(durVal).toFixed(2)}s` : "--";
            const inSize = s.input_size !== undefined && s.input_size !== null ? s.input_size : "--";
            const outSize = s.output_size !== undefined && s.output_size !== null ? s.output_size : (s.chunk_count || s.node_count || "--");
            const statusClass = s.status === "completed" ? "completed" : (s.status === "fallback" ? "fallback" : "pending");
            return `
                <tr>
                    <td><strong>${escapeHtml(name)}</strong></td>
                    <td><span class="stage-status-pill ${statusClass}">${escapeHtml(s.status || "pending")}</span></td>
                    <td><span class="confidence-score">${dur}</span></td>
                    <td>${inSize}</td>
                    <td>${outSize}</td>
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

function connectLogStream() {
    if (logSocket) {
        logSocket.close();
        logSocket = null;
    }
    stopLogPolling();

    try {
        logSocket = new WebSocket(`${WS_BASE}/logs/stream`);
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
    logPollTimer = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/logs/recent?lines=80`);
            if (response.ok) {
                const payload = await response.json();
                const consoleBox = document.getElementById("logs-console");
                if (payload.logs && Array.isArray(payload.logs)) {
                    consoleBox.textContent = payload.logs.join("\n");
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
    const line = typeof payload === "string" ? payload : `[${payload.level || "INFO"}] ${payload.message || JSON.stringify(payload)}`;
    consoleBox.textContent += line + "\n";
    consoleBox.scrollTop = consoleBox.scrollHeight;
    trimLogs(consoleBox);
}

function appendRawLog(raw) {
    const consoleBox = document.getElementById("logs-console");
    consoleBox.textContent += raw + "\n";
    consoleBox.scrollTop = consoleBox.scrollHeight;
    trimLogs(consoleBox);
}

function trimLogs(consoleBox) {
    const lines = consoleBox.textContent.split("\n");
    if (lines.length > 300) {
        consoleBox.textContent = lines.slice(-250).join("\n");
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
                        if (type === "subtopic") return "#6366f1";
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
                        return ele.data("relation_type") === "hierarchy" ? "#475569" : "#8b5cf6";
                    },
                    "target-arrow-color": function (ele) {
                        return ele.data("relation_type") === "hierarchy" ? "#475569" : "#8b5cf6";
                    },
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    opacity: 0.6,
                },
            },
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
        ],
        layout: {
            name: "preset",
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
        const definition = data.definition || data.description || "Identified concept in document knowledge map.";
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
            <strong>${escapeHtml(data.relation_type || "relates-to")}</strong><br>
            <div style="font-size: 0.78rem; margin-top: 4px; color: #cbd5e1">${escapeHtml(data.description || "Semantic candidate relation")}</div>
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

    // Inspector click handler
    graph.on("tap", "node", function (e) {
        inspectNode(e.target.data());
    });
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

    const definition = (matchedConcept && matchedConcept.definition) || data.definition || data.description || "No formal definition provided in chunk context.";
    const occurrences = (matchedConcept && matchedConcept.frequency) ? `${matchedConcept.frequency} occurrences` : (data.frequency ? `${data.frequency} occurrences` : "Identified in graph synthesis");
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
                    ${connected.slice(0, 4).map(r => `
                        <div class="relation-item-pill">
                            <span class="relation-type-tag">${escapeHtml(r.relation_type)}</span>
                            <span>${escapeHtml(r.from_concept)} ➔ ${escapeHtml(r.to_concept)}</span>
                            <span class="confidence-score">${Number(r.confidence || 0).toFixed(2)}</span>
                        </div>
                    `).join("")}
                </div>
            `;
        }
    }

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
        ${relationsHtml}
    `;
}

function filterGraphSearch(query) {
    if (!graph) return;
    const cleanQuery = query.trim().toLowerCase();
    graph.nodes().removeClass("highlighted");

    if (!cleanQuery) return;

    const matches = graph.nodes().filter(node => {
        const label = String(node.data("label") || node.data("name") || "").toLowerCase();
        return label.includes(cleanQuery);
    });

    matches.addClass("highlighted");
    if (matches.length > 0) {
        graph.center(matches);
    }
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
        graph.layout({
            name: "preset",
            positions: buildMindMapPositions(elements),
            animate: true,
            fit: true,
            padding: 50,
        }).run();
        graph.autoungrabify(true);

        const nodeCount = graph.nodes().length;
        const edgeCount = graph.edges().length;
        document.getElementById("graph-info").textContent = `${nodeCount} nodes / ${edgeCount} edges`;

        updateOverviewMetric("stat-nodes", nodeCount);
        updateOverviewMetric("stat-edges", edgeCount);

        applyRelationFilter(document.getElementById("relation-filter")?.value || "all");
    } catch (error) {
        document.getElementById("graph-info").textContent = "Graph load failed.";
    }
}

function resetGraphPanels(message) {
    document.getElementById("graph-info").textContent = message;
    if (graph) graph.elements().remove();
}

function applyRelationFilter(filterVal) {
    if (!graph) return;
    graph.batch(() => {
        if (filterVal === "all") {
            graph.edges().style("display", "element");
        } else if (filterVal === "semantic") {
            graph.edges().forEach((e) => {
                const rel = e.data("relation_type");
                e.style("display", rel !== "hierarchy" ? "element" : "none");
            });
        } else if (filterVal === "hierarchy") {
            graph.edges().forEach((e) => {
                const rel = e.data("relation_type");
                e.style("display", rel === "hierarchy" ? "element" : "none");
            });
        }
    });
}

function simplifyGraphElements(elementsArr) {
    return elementsArr;
}

function buildMindMapPositions(elementsArr) {
    const nodes = elementsArr.filter((e) => e.data && e.data.source === undefined);
    const positions = {};
    const radius = Math.max(160, nodes.length * 16);
    nodes.forEach((n, idx) => {
        const angle = (2 * Math.PI * idx) / Math.max(nodes.length, 1);
        positions[n.data.id] = {
            x: Math.round(radius * Math.cos(angle)),
            y: Math.round(radius * Math.sin(angle)),
        };
    });
    return positions;
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
            } else if (!llmAvailable || llmCheck.status === "unhealthy") {
                updateHonestStatus("LLM Degraded", "degraded");
            } else {
                updateHonestStatus("LLM Ready", "ready");
            }
        }
    } catch {
        const dot = document.getElementById("system-dot");
        if (dot) dot.className = "status-indicator-dot error";
        document.getElementById("health-pill").textContent = "System: offline";
        updateHonestStatus("LLM Degraded", "degraded");
    }
}

function isDocumentAnalyzed(doc) {
    return String(doc?.status || "").toLowerCase() === "analyzed";
}

function renderDatasetStats(stats) {
    const box = document.getElementById("dataset-stats");
    if (!box) return;
    box.innerHTML = `
        <strong>Export Records:</strong> ${stats.total_records || 0}<br>
        <strong>Status:</strong> ${escapeHtml(stats.exportFolder || "Ready")}<br>
        <strong>Latest File:</strong> ${escapeHtml(stats.latestExport || "None")}
    `;
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
    const pill = document.getElementById("llm-pill");

    if (badge) badge.textContent = statusText;
    if (chip) {
        chip.className = `status-chip honest-${variant || "ready"}`;
    }
    if (pill) {
        pill.textContent = statusText;
        pill.className = `status-pill ${variant || "ready"}`;
    }
}
