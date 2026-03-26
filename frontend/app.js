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
    { key: "ingestion", label: "Upload" },
    { key: "chunking", label: "Chunking" },
    { key: "embedding", label: "Embedding" },
    { key: "concept_extraction", label: "Concept Extraction" },
    { key: "relation_extraction", label: "Relation Extraction" },
    { key: "graph_building", label: "Graph Building" },
    { key: "evaluation", label: "Evaluation" },
    { key: "hierarchy", label: "Hierarchy" },
];
const STAGE_INDEX = Object.fromEntries(PIPELINE_STAGES.map((stage, index) => [stage.key, index]));

let selectedDocId = null;
let selectedDoc = null;
let graph = null;
let logSocket = null;
let monitorTimer = null;
let logPollTimer = null;

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
        if (graph) graph.zoom(graph.zoom() * 1.2);
    });
    document.getElementById("zoom-out").addEventListener("click", () => {
        if (graph) graph.zoom(graph.zoom() * 0.85);
    });
    document.getElementById("fit-graph").addEventListener("click", () => {
        if (graph) graph.fit(undefined, 30);
    });
}

async function uploadDocument(event) {
    event.preventDefault();
    const fileInput = document.getElementById("file-input");
    const file = fileInput.files[0];
    if (!file) {
        setStatus("upload-status", "Select a file first.", "error");
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
            item.innerHTML = `
                <div class="name">${doc.name}</div>
                <div class="meta">${doc.status} - ${(doc.size / 1024 / 1024).toFixed(2)} MB</div>
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

function selectDocument(doc) {
    selectedDocId = doc.id;
    selectedDoc = doc;
    document.querySelectorAll(".document-item").forEach((item) => item.classList.remove("selected"));
    const matching = document.querySelector(`.document-item[data-doc-id="${selectedDocId}"]`);
    if (matching) matching.classList.add("selected");

    document.getElementById("doc-info").innerHTML = `
        <strong>${doc.name}</strong><br>
        Status: ${doc.status}<br>
        Size: ${(doc.size / 1024 / 1024).toFixed(2)} MB
    `;

    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Start Analysis";

    resetPipelineBoard();
    connectLogStream();
    if (isDocumentAnalyzed(doc)) {
        loadGraph(doc.id);
        loadAnalysisReport();
    } else {
        resetGraphPanels("Graph will appear after analysis completes.");
        document.getElementById("analysis-report").textContent = "Analysis report will appear after processing completes.";
        renderDatasetStats({
            total_records: 0,
            labeled_records: 0,
            unlabeled_records: 0,
            exportFolder: "Pending analysis",
            latestExport: "No export yet",
        });
    }
    refreshPipelineStatus();
    monitorPipeline();
}

async function startAnalysis() {
    if (!selectedDocId) return;
    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Submitting...";
    setStatus("analysis-status", "Submitting pipeline job...", "info");
    document.getElementById("analysis-report").textContent = "Analysis in progress. Summary and findings will appear when processing completes.";
    renderDatasetStats({
        total_records: 0,
        labeled_records: 0,
        unlabeled_records: 0,
        exportFolder: "Export will appear after analysis",
        latestExport: "Generating dataset export",
    });
    resetPipelineBoard();

    try {
        const response = await fetch(`${API_BASE}/analyze/${selectedDocId}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                generate_hierarchy: true,
                run_evaluation: true,
                force_reanalyze: String(selectedDoc?.status || "").toLowerCase() === "analyzed",
            }),
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Failed to start analysis");
        }
        setStatus(
            "analysis-status",
            payload.status === "already_analyzed"
                ? "Document already analyzed."
                : `Pipeline started. Job: ${payload.job_id}`,
            "success"
        );
        analyzeBtn.textContent = "Running...";
        monitorPipeline();
    } catch (error) {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Start Analysis";
        setStatus("analysis-status", `Analysis error: ${error.message}`, "error");
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
    }, 1200);
}

async function refreshPipelineStatus() {
    if (!selectedDocId) return;
    let usedLegacyFallback = false;
    try {
        const response = await fetch(
            `${API_BASE}/system/pipeline-status?document_id=${encodeURIComponent(selectedDocId)}`
        );
        if (response.ok) {
            const payload = await response.json();
            if (!payload || payload.status === "not_found") {
                usedLegacyFallback = true;
            } else {
                renderPipelineStatus(payload);

                const summary = document.getElementById("pipeline-summary");
                summary.textContent = `${payload.status} - ${payload.overall_progress || 0}%`;

                if (payload.status === "completed" || payload.status === "failed") {
                    const analyzeBtn = document.getElementById("analyze-btn");
                    analyzeBtn.disabled = false;
                    analyzeBtn.textContent = "Start Analysis";
                    if (payload.status === "completed") {
                        setStatus("analysis-status", "Pipeline completed.", "success");
                        await loadDocuments();
                        await loadGraph(selectedDocId);
                    }
                    if (payload.status === "failed") {
                        setStatus("analysis-status", `Pipeline failed: ${payload.error || "unknown error"}`, "error");
                    }
                    clearInterval(monitorTimer);
                    monitorTimer = null;
                }
                return;
            }
        } else if (response.status === 404) {
            usedLegacyFallback = true;
        }
    } catch (error) {
        usedLegacyFallback = true;
    }

    if (usedLegacyFallback) {
        await refreshLegacyPipelineStatus();
    }
}

async function refreshLegacyPipelineStatus() {
    if (!selectedDocId) return;
    try {
        const response = await fetch(`${API_BASE}/status/${encodeURIComponent(selectedDocId)}`);
        if (!response.ok) return;

        const payload = await response.json();
        renderLegacyPipelineStatus(payload);
    } catch (error) {
        // Ignore legacy fallback errors to avoid noisy UI.
    }
}

function normalizeLegacyStage(stage) {
    const value = String(stage || "").toLowerCase();
    if (value.includes("ingest")) return "ingestion";
    if (value.includes("chunk")) return "chunking";
    if (value.includes("embed")) return "embedding";
    if (value.includes("concept")) return "concept_extraction";
    if (value.includes("relation")) return "relation_extraction";
    if (value.includes("map")) return "relation_extraction";
    if (value.includes("graph")) return "graph_building";
    if (value.includes("evaluat")) return "evaluation";
    if (value.includes("hierarchy")) return "hierarchy";
    return "";
}

function renderLegacyPipelineStatus(payload) {
    const summary = document.getElementById("pipeline-summary");
    const currentJob = payload.current_job;
    const documentStatus = String(payload.document_status || "").toLowerCase();

    resetPipelineBoard();

    if (currentJob) {
        const normalizedStage = normalizeLegacyStage(currentJob.current_stage);
        const stageKeys = PIPELINE_STAGES.map((stage) => stage.key);
        const stageIndex = stageKeys.indexOf(normalizedStage);

        PIPELINE_STAGES.forEach((stage, index) => {
            const fill = document.getElementById(`fill-${stage.key}`);
            const status = document.getElementById(`status-${stage.key}`);

            if (stageIndex >= 0 && index < stageIndex) {
                fill.style.width = "100%";
                status.textContent = "completed";
            } else if (stageIndex >= 0 && index === stageIndex) {
                const approx = Math.max(10, Math.min(95, Number(currentJob.progress_percent || 0)));
                fill.style.width = `${approx}%`;
                status.textContent = currentJob.status || "running";
            } else {
                fill.style.width = "0%";
                status.textContent = "pending";
            }
        });

        summary.textContent = `Legacy status endpoint: ${currentJob.status} - ${currentJob.current_stage} (${currentJob.progress_percent}%)`;
    } else if (documentStatus === "analyzed") {
        PIPELINE_STAGES.forEach((stage) => {
            document.getElementById(`fill-${stage.key}`).style.width = "100%";
            document.getElementById(`status-${stage.key}`).textContent = "completed";
        });
        summary.textContent = "Completed (legacy status endpoint)";
    } else {
        summary.textContent = "Waiting for execution (legacy status endpoint)";
    }

    const jobStatus = String(currentJob?.status || "").toLowerCase();
    const completed = documentStatus === "analyzed" || jobStatus === "completed";
    const failed = jobStatus === "failed" || jobStatus === "cancelled";

    if (completed || failed) {
        const analyzeBtn = document.getElementById("analyze-btn");
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Start Analysis";

        if (completed) {
            setStatus("analysis-status", "Pipeline completed.", "success");
            loadDocuments().then(() => loadGraph(selectedDocId));
        } else if (failed) {
            setStatus("analysis-status", `Pipeline failed: ${currentJob?.error_message || "unknown error"}`, "error");
        }

        if (monitorTimer) {
            clearInterval(monitorTimer);
            monitorTimer = null;
        }
    }
}

function initPipelineBoard() {
    const container = document.getElementById("pipeline-stages");
    container.innerHTML = "";
    PIPELINE_STAGES.forEach((stage) => {
        const row = document.createElement("div");
        row.className = "stage-row";
        row.id = `stage-${stage.key}`;
        row.innerHTML = `
            <div class="stage-head">
                <span>${stage.label}</span>
                <span class="stage-status" id="status-${stage.key}">pending</span>
            </div>
            <div class="stage-bar">
                <div class="stage-fill" id="fill-${stage.key}"></div>
            </div>
        `;
        container.appendChild(row);
    });
}

function resetPipelineBoard() {
    PIPELINE_STAGES.forEach((stage) => {
        const fill = document.getElementById(`fill-${stage.key}`);
        const status = document.getElementById(`status-${stage.key}`);
        fill.style.width = "0%";
        status.textContent = "pending";
    });
    document.getElementById("pipeline-summary").textContent = "Waiting for execution";
}

function renderPipelineStatus(payload) {
    const stages = payload.stages || {};
    PIPELINE_STAGES.forEach((stage) => {
        let stagePayload = stages[stage.key] || {};
        if (!stagePayload.status && (stage.key === "concept_extraction" || stage.key === "relation_extraction")) {
            stagePayload = stages["mapping"] || {};
        }
        const fill = document.getElementById(`fill-${stage.key}`);
        const status = document.getElementById(`status-${stage.key}`);
        const progress =
            stagePayload.progress != null
                ? stagePayload.progress
                : stagePayload.status === "completed"
                    ? 100
                    : 0;
        fill.style.width = `${Math.max(0, Math.min(100, progress))}%`;
        status.textContent = stagePayload.status || "pending";
    });
}

function setStageState(stageKey, stageStatus, progress) {
    const fill = document.getElementById(`fill-${stageKey}`);
    const status = document.getElementById(`status-${stageKey}`);
    if (!fill || !status) return;
    fill.style.width = `${Math.max(0, Math.min(100, Number(progress || 0)))}%`;
    status.textContent = stageStatus || "pending";
}

function applyStageEventFromLog(payload) {
    let stageKey = payload.stage;
    if (stageKey === "mapping") {
        _applyLogEventToKey("concept_extraction", payload);
        _applyLogEventToKey("relation_extraction", payload);
        return;
    }
    _applyLogEventToKey(stageKey, payload);
}

function _applyLogEventToKey(stageKey, payload) {
    const stagePos = STAGE_INDEX[stageKey];
    if (stagePos == null) return;

    const eventName = String(payload.event || "").toLowerCase();
    const stageLabel = PIPELINE_STAGES[stagePos]?.label || stageKey;
    const summary = document.getElementById("pipeline-summary");

    if (eventName === "stage_start") {
        for (let i = 0; i < stagePos; i += 1) {
            const prevKey = PIPELINE_STAGES[i].key;
            const prevStatus = document.getElementById(`status-${prevKey}`)?.textContent || "pending";
            if (prevStatus === "pending" || prevStatus === "running") {
                setStageState(prevKey, "completed", 100);
            }
        }
        setStageState(stageKey, "running", 35);
        summary.textContent = `running - ${stageLabel}`;
        return;
    }

    if (eventName === "stage_complete" || eventName === "stage_skipped") {
        setStageState(stageKey, eventName === "stage_complete" ? "completed" : "skipped", 100);
        summary.textContent = `running - ${stageLabel} done`;
        return;
    }

    if (eventName === "stage_error") {
        setStageState(stageKey, "failed", 100);
        summary.textContent = `failed - ${stageLabel}`;
    }
}

function connectLogStream() {
    if (!selectedDocId) return;
    if (logSocket) {
        logSocket.onclose = null;
        logSocket.close();
    }
    stopLogPolling();
    const wsUrl = `${WS_BASE}/logs/stream?document_id=${encodeURIComponent(selectedDocId)}`;
    let opened = false;
    logSocket = new WebSocket(wsUrl);

    logSocket.onopen = () => {
        opened = true;
    };

    logSocket.onmessage = (event) => {
        try {
            const payload = JSON.parse(event.data);
            appendLogLine(payload);
        } catch {
            appendRawLog(event.data);
        }
    };

    logSocket.onclose = () => {
        if (!selectedDocId) return;
        if (!opened) {
            appendRawLog("[warning] Live websocket logs unavailable. Falling back to /logs/recent polling.");
            startLogPolling();
            return;
        }
        if (selectedDocId) {
            setTimeout(connectLogStream, 1200);
        }
    };
}

function startLogPolling() {
    if (logPollTimer || !selectedDocId) return;
    logPollTimer = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/logs/recent?lines=80`);
            if (!response.ok) return;
            const payload = await response.json();
            const lines = payload.lines || [];
            const consoleBox = document.getElementById("logs-console");
            consoleBox.textContent = lines.join("\n");
            consoleBox.scrollTop = consoleBox.scrollHeight;
        } catch (error) {
            // Ignore intermittent polling failures.
        }
    }, 2000);
}

function stopLogPolling() {
    if (!logPollTimer) return;
    clearInterval(logPollTimer);
    logPollTimer = null;
}

function appendLogLine(payload) {
    const consoleBox = document.getElementById("logs-console");
    const stage = payload.stage || "system";
    const event = payload.event || "log";
    const message = payload.message || "";

    let colorClass = "";
    const eventLower = event.toLowerCase();
    const messageLower = message.toLowerCase();

    if (eventLower.includes("error") || messageLower.includes("error")) colorClass = "log-error";
    else if (eventLower.includes("warn") || messageLower.includes("warn")) colorClass = "log-warning";
    else if (eventLower === "stage_start") colorClass = "log-stage-start";
    else if (eventLower === "stage_complete") colorClass = "log-stage-complete";
    else colorClass = "log-info";

    const lineSpan = document.createElement("span");
    lineSpan.className = colorClass;
    lineSpan.textContent = `[${stage}] [${event}] ${message}\n`;
    consoleBox.appendChild(lineSpan);

    trimLogs(consoleBox);
    applyStageEventFromLog(payload);
}

function appendRawLog(raw) {
    const consoleBox = document.getElementById("logs-console");
    const lineSpan = document.createElement("span");
    lineSpan.className = "log-info";
    lineSpan.textContent = `${raw}\n`;
    consoleBox.appendChild(lineSpan);
    trimLogs(consoleBox);
}

function trimLogs(consoleBox) {
    while (consoleBox.children.length > 500) {
        consoleBox.removeChild(consoleBox.firstChild);
    }
    consoleBox.scrollTop = consoleBox.scrollHeight;
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
                        const type = ele.data('type');
                        if (type === 'topic') return '#0ea5a4';
                        if (type === 'subtopic') return '#3b82f6';
                        if (type === 'document') return '#f97316';
                        return '#1d4ed8';
                    },
                    shape: function (ele) {
                        const type = ele.data('type');
                        if (type === 'document') return 'round-rectangle';
                        return 'round-rectangle';
                    },
                    label: function (ele) {
                        return graphDisplayLabel(ele.data());
                    },
                    color: "#f8fafc",
                    "text-valign": "center",
                    "text-halign": "center",
                    "text-wrap": "wrap",
                    "text-max-width": function (ele) {
                        const type = ele.data('type');
                        if (type === 'document') return 135;
                        if (type === 'topic') return 150;
                        if (type === 'subtopic') return 145;
                        return 140;
                    },
                    "font-size": function (ele) {
                        const type = ele.data('type');
                        if (type === 'document') return '13px';
                        if (type === 'topic') return '12px';
                        if (type === 'subtopic') return '11px';
                        return '10px';
                    },
                    "font-weight": function (ele) {
                        return (ele.data('type') === 'document' || ele.data('type') === 'topic') ? '700' : '600';
                    },
                    width: function (ele) {
                        const type = ele.data('type');
                        if (type === 'document') return 150;
                        if (type === 'topic') return 168;
                        if (type === 'subtopic') return 158;
                        return 150;
                    },
                    height: function (ele) {
                        const type = ele.data('type');
                        if (type === 'document') return 52;
                        if (type === 'topic') return 48;
                        if (type === 'subtopic') return 46;
                        return 44;
                    },
                    "border-width": 2,
                    "border-color": function (ele) {
                        const type = ele.data('type');
                        if (type === 'document') return '#ea580c';
                        if (type === 'topic') return '#0f766e';
                        if (type === 'subtopic') return '#2563eb';
                        return '#1e40af';
                    },
                    padding: "8px",
                    "overlay-padding": "6px"
                },
            },
            {
                selector: "edge",
                style: {
                    width: function (ele) {
                        return ele.data('relation_type') === 'hierarchy' ? 2.2 : 1.2;
                    },
                    "line-color": function (ele) {
                        return ele.data('relation_type') === 'hierarchy' ? '#475569' : '#64748b';
                    },
                    "target-arrow-color": function (ele) {
                        return ele.data('relation_type') === 'hierarchy' ? '#475569' : '#64748b';
                    },
                    "target-arrow-shape": "triangle",
                    "curve-style": "bezier",
                    opacity: function (ele) {
                        return ele.data('relation_type') === 'hierarchy' ? 0.9 : 0.45;
                    },
                    label: "",
                },
            },
            {
                selector: 'edge[relation_type = "hierarchy"]',
                style: {
                    "target-arrow-shape": "none",
                },
            },
        ],
        layout: {
            name: "preset",
            padding: 50,
            animate: true,
            fit: true,
        },
        autoungrabify: true,
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false
    });

    graph.on('mouseover', 'node', function (e) {
        const data = e.target.data();
        const tooltip = document.getElementById("graph-tooltip");
        const nodeType = String(data.type || "node").replaceAll("-", " ");
        const centrality = Number(data.centrality ?? data.pagerank ?? 0).toFixed(3);
        const definition = data.definition || data.description || "No definition available.";
        tooltip.innerHTML = `<strong>${escapeHtml(data.label)}</strong><br>
                             <div>Type: ${escapeHtml(nodeType)}</div>
                             <div>Importance: ${escapeHtml(centrality)}</div>
                             <div style="font-size: 0.75rem; margin-top: 6px; color: #cbd5e1">${escapeHtml(definition)}</div>`;
        tooltip.style.display = "block";
    });

    graph.on('mouseover', 'edge', function (e) {
        const data = e.target.data();
        const tooltip = document.getElementById("graph-tooltip");
        const relationLabel = graphRelationLabel(data.relation_type);
        const relationDescription =
            data.description ||
            (String(data.relation_type || "").toLowerCase() === "explanation"
                ? "Related concepts found together in the same source sections."
                : "Connection between nodes in the knowledge map.");
        tooltip.innerHTML = `<strong>${escapeHtml(relationLabel)}</strong><br>
                             <div style="font-size: 0.75rem; margin-top: 6px; color: #cbd5e1">${escapeHtml(relationDescription)}</div>`;
        tooltip.style.display = "block";
    });

    graph.on('mousemove', function (e) {
        const tooltip = document.getElementById("graph-tooltip");
        if (tooltip.style.display === "block") {
            // position slightly near the cursor
            tooltip.style.left = (e.originalEvent.pageX + 10) + 'px';
            tooltip.style.top = (e.originalEvent.pageY + 10) + 'px';
        }
    });

    graph.on('mouseout', 'node, edge', function () {
        document.getElementById("graph-tooltip").style.display = "none";
    });
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
        const semanticEdgeCount = graph.edges().filter(edge => edge.data('relation_type') !== 'hierarchy').length;
        document.getElementById("graph-info").textContent = `${nodeCount} nodes / ${edgeCount} edges (${semanticEdgeCount} semantic)`;
        populateGraphFilterAndTable(elements);
        applyRelationFilter(document.getElementById("relation-filter")?.value || "hierarchy");
    } catch (error) {
        document.getElementById("graph-info").textContent = "Graph load failed.";
    }
}

function resetGraphPanels(message) {
    document.getElementById("graph-info").textContent = message;
    graph.elements().remove();
    populateGraphFilterAndTable([]);
}

function populateGraphFilterAndTable(elementsArr) {
    const nodes = elementsArr.filter(e => e.data && (e.data.source === undefined));
    const edges = elementsArr.filter(e => e.data && e.data.source !== undefined);
    const structuralLabels = new Set(
        nodes
            .filter((node) => ["document", "topic", "subtopic"].includes(String(node.data?.type || "").toLowerCase()))
            .map((node) => normalizeConceptLabel(node.data?.label || node.data?.name || ""))
            .filter(Boolean)
    );
    const conceptNodes = nodes
        .filter((node) => String(node.data?.type || "").toLowerCase() === "concept")
        .filter((node) => !structuralLabels.has(normalizeConceptLabel(node.data?.label || node.data?.name || "")))
        .filter((node) => isDisplayableConcept(node.data));

    const tbody = document.querySelector("#concepts-table tbody");
    tbody.innerHTML = "";

    let sortedNodes = [...conceptNodes].sort((a, b) => (
        Number(b.data?.centrality ?? b.data?.pagerank ?? 0) - Number(a.data?.centrality ?? a.data?.pagerank ?? 0)
    ));
    if (sortedNodes.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="muted" style="text-align: center;">No concepts loaded</td></tr>`;
    } else {
        const html = sortedNodes.slice(0, 100).map(n => {
            const data = n.data;
            const label = data.label || data.id;
            const desc = displayableConceptDefinition(data);
            return `<tr><td>${escapeHtml(label)}</td><td>${escapeHtml(desc)}</td></tr>`;
        }).join("");
        tbody.innerHTML = html;
    }
    document.getElementById("concepts-count").textContent = `${sortedNodes.length} concepts`;

    const edgeTypes = new Set();
    edges.forEach(e => {
        if (e.data.relation_type) edgeTypes.add(e.data.relation_type);
    });
    const filter = document.getElementById("relation-filter");
    const currentValue = filter.value || "hierarchy";
    filter.innerHTML = `
        <option value="hierarchy">Structure Only</option>
        <option value="all">All Relations</option>
        <option value="semantic">Semantic Only</option>
    `;
    [...edgeTypes].sort().forEach(t => {
        if (t === "hierarchy") return;
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        filter.appendChild(opt);
    });
    filter.value = [...filter.options].some((option) => option.value === currentValue) ? currentValue : "hierarchy";
}

function normalizeConceptLabel(value) {
    return String(value || "").trim().toLowerCase().replace(/\s+/g, " ");
}

function isDisplayableConcept(data) {
    const label = String(data?.label || data?.name || "").trim();
    const normalized = normalizeConceptLabel(label);
    if (!label) return false;
    if (label.length > 50) return false;
    if (/\d/.test(label)) return false;
    if (/^(document|question|questions|page|attempt|part|section)$/i.test(label)) return false;
    if (/page\s+\d+/i.test(label)) return false;
    if (/attempt\s+\d+/i.test(label)) return false;
    if (/(check the schedule|semester examination|end semester|dashboard|reporting)/i.test(label)) return false;

    const words = normalized.match(/[a-z][a-z'-]*/g) || [];
    if (!words.length || words.length > 4) return false;
    const noiseWords = new Set([
        "the", "and", "for", "with", "from", "into", "page", "attempt", "question", "questions",
        "part", "section", "check", "schedule", "semester", "examination", "reporting",
    ]);
    const meaningfulWords = words.filter((word) => !noiseWords.has(word));
    if (!meaningfulWords.length) return false;
    if (words.length >= 3 && meaningfulWords.length < words.length - 1) return false;
    return true;
}

function displayableConceptDefinition(data) {
    const raw = String(data?.definition || data?.description || "").trim();
    if (!raw) return "Source concept identified from document text.";
    if (/frequent concept inferred from document context/i.test(raw)) {
        return "Source concept identified from document text.";
    }
    return raw;
}

function applyRelationFilter(filterVal) {
    if (!graph) return;
    graph.edges().forEach((edge) => {
        const relationType = String(edge.data('relation_type') || '').toLowerCase();
        let visible = false;
        if (filterVal === "all") {
            visible = true;
        } else if (filterVal === "hierarchy") {
            visible = relationType === "hierarchy";
        } else if (filterVal === "semantic") {
            visible = relationType !== "hierarchy";
        } else {
            visible = relationType === String(filterVal || "").toLowerCase();
        }
        edge.style('display', visible ? 'element' : 'none');
    });
}

function simplifyGraphElements(elementsArr) {
    const nodes = elementsArr.filter((item) => item.data && item.data.source === undefined);
    const edges = elementsArr.filter((item) => item.data && item.data.source !== undefined);
    const nodeMap = new Map(nodes.map((node) => [String(node.data.id), node]));
    const hierarchyEdges = edges.filter((edge) => String(edge.data?.relation_type || "").toLowerCase() === "hierarchy");
    const incomingHierarchy = new Map();
    const outgoingHierarchy = new Map();

    hierarchyEdges.forEach((edge) => {
        const source = String(edge.data.source);
        const target = String(edge.data.target);
        if (!outgoingHierarchy.has(source)) outgoingHierarchy.set(source, []);
        if (!incomingHierarchy.has(target)) incomingHierarchy.set(target, []);
        outgoingHierarchy.get(source).push(edge);
        incomingHierarchy.get(target).push(edge);
    });

    const nodesToRemove = new Set();
    const edgesToRemove = new Set();
    const edgesToAdd = [];
    const replacementKeys = new Set();

    nodes.forEach((node) => {
        const data = node.data || {};
        const nodeType = String(data.type || "").toLowerCase();
        if (!["topic", "subtopic"].includes(nodeType)) return;

        const outgoing = outgoingHierarchy.get(String(data.id)) || [];
        if (outgoing.length !== 1) return;

        const childEdge = outgoing[0];
        const childNode = nodeMap.get(String(childEdge.data.target));
        if (!childNode || String(childNode.data?.type || "").toLowerCase() !== "concept") return;

        const nodeLabel = normalizeConceptLabel(data.label || data.name || "");
        const childLabel = normalizeConceptLabel(childNode.data?.label || childNode.data?.name || "");
        if (!nodeLabel || nodeLabel !== childLabel) return;

        nodesToRemove.add(String(data.id));
        edgesToRemove.add(String(childEdge.data.id));

        const incoming = incomingHierarchy.get(String(data.id)) || [];
        incoming.forEach((edge) => {
            edgesToRemove.add(String(edge.data.id));
            const sourceId = String(edge.data.source);
            const targetId = String(childNode.data.id);
            const replacementKey = `${sourceId}->${targetId}`;
            if (sourceId === targetId || replacementKeys.has(replacementKey)) return;
            replacementKeys.add(replacementKey);
            edgesToAdd.push({
                data: {
                    id: `collapsed-${sourceId}-${targetId}`,
                    source: sourceId,
                    target: targetId,
                    relation_type: "hierarchy",
                    description: "Collapsed duplicate hierarchy wrapper.",
                },
            });
        });
    });

    const keptNodes = nodes.filter((node) => !nodesToRemove.has(String(node.data.id)));
    const keptEdges = edges.filter((edge) => !edgesToRemove.has(String(edge.data.id)));
    return [...keptNodes, ...keptEdges, ...edgesToAdd];
}

function buildMindMapPositions(elementsArr) {
    const nodes = elementsArr.filter((item) => item.data && item.data.source === undefined);
    const edges = elementsArr.filter((item) => item.data && item.data.source !== undefined);
    const hierarchyEdges = edges.filter((edge) => String(edge.data?.relation_type || "").toLowerCase() === "hierarchy");
    const childMap = new Map();
    const nodeMap = new Map();

    nodes.forEach((node) => {
        nodeMap.set(String(node.data.id), node.data);
        childMap.set(String(node.data.id), []);
    });

    hierarchyEdges.forEach((edge) => {
        const source = String(edge.data.source);
        const target = String(edge.data.target);
        if (!childMap.has(source)) {
            childMap.set(source, []);
        }
        childMap.get(source).push(target);
    });

    const sortChildren = (leftId, rightId) => {
        const left = nodeMap.get(leftId) || {};
        const right = nodeMap.get(rightId) || {};
        const leftType = layoutTypeOrder(String(left.type || ""));
        const rightType = layoutTypeOrder(String(right.type || ""));
        if (leftType !== rightType) {
            return leftType - rightType;
        }
        return String(left.label || "").localeCompare(String(right.label || ""));
    };

    childMap.forEach((children, key) => {
        children.sort(sortChildren);
        childMap.set(key, children);
    });

    const positions = {};
    const visited = new Set();
    const horizontalSpacing = 190;
    const verticalSpacing = 130;
    let leafCursor = 0;

    const rootNode = nodes.find((node) => String(node.data?.type || "").toLowerCase() === "document");
    const rootId = rootNode ? String(rootNode.data.id) : (nodes[0] ? String(nodes[0].data.id) : null);

    const placeNode = (nodeId, depth) => {
        visited.add(nodeId);
        const children = (childMap.get(nodeId) || []).filter((childId) => nodeMap.has(childId));
        const y = 90 + (depth * verticalSpacing);
        if (!children.length) {
            const x = 120 + (leafCursor * horizontalSpacing);
            leafCursor += 1;
            positions[nodeId] = { x, y };
            return x;
        }

        const childXs = children.map((childId) => placeNode(childId, depth + 1));
        const x = (Math.min(...childXs) + Math.max(...childXs)) / 2;
        positions[nodeId] = { x, y };
        return x;
    };

    if (rootId) {
        placeNode(rootId, 0);
    }

    const orphanNodes = nodes
        .map((node) => String(node.data.id))
        .filter((id) => !visited.has(id))
        .sort(sortChildren);

    orphanNodes.forEach((nodeId) => {
        const data = nodeMap.get(nodeId) || {};
        const type = String(data.type || "").toLowerCase();
        const depth = type === "topic" ? 1 : type === "subtopic" ? 2 : type === "concept" ? 3 : 0;
        positions[nodeId] = {
            x: 120 + (leafCursor * horizontalSpacing),
            y: 90 + (depth * verticalSpacing),
        };
        leafCursor += 1;
    });

    return positions;
}

function layoutTypeOrder(type) {
    const normalized = String(type || "").toLowerCase();
    if (normalized === "document") return 0;
    if (normalized === "topic") return 1;
    if (normalized === "subtopic") return 2;
    if (normalized === "concept") return 3;
    return 4;
}

function graphDisplayLabel(data) {
    const label = String(data?.label || data?.name || "").trim();
    const type = String(data?.type || data?.node_type || "").toLowerCase();
    if (!label) return "";

    let maxLength = 22;
    if (type === "document") maxLength = 18;
    else if (type === "topic") maxLength = 24;
    else if (type === "subtopic") maxLength = 22;
    else if (type === "concept") maxLength = 18;

    const compact = label.replace(/\s+/g, " ").trim();
    if (compact.length <= maxLength) {
        return compact;
    }
    return `${compact.slice(0, Math.max(0, maxLength - 1)).trim()}…`;
}

function graphRelationLabel(value) {
    const normalized = String(value || "").trim().toLowerCase();
    if (!normalized) return "Relation";
    if (normalized === "hierarchy") return "Hierarchy";
    if (normalized === "explanation") return "Related Concepts";
    return normalized
        .split("-")
        .map((part) => part ? part[0].toUpperCase() + part.slice(1) : "")
        .join(" ");
}

function normalizeGraphNode(node, index) {
    const raw = node?.data ? { ...node.data } : { ...(node || {}) };
    const normalized = {
        id: String(raw.id || `node-${index}`),
        label: raw.label || raw.name || String(raw.id || `node-${index}`),
        type: raw.type || raw.node_type || "unknown",
        centrality: raw.centrality ?? raw.pagerank,
        pagerank: raw.pagerank,
        community: raw.community,
        definition: raw.definition || raw.description,
        ...raw,
    };
    normalized.id = String(normalized.id);
    normalized.label = normalized.label || normalized.name || normalized.id;
    normalized.type = normalized.type || normalized.node_type || "unknown";
    normalized.centrality = normalized.centrality ?? normalized.pagerank;
    return { data: normalized };
}

function normalizeGraphEdge(edge, index) {
    const raw = edge?.data ? { ...edge.data } : { ...(edge || {}) };
    const normalized = {
        id: String(raw.id || `edge-${index}`),
        source: String(raw.source || ""),
        target: String(raw.target || ""),
        relation_type: raw.relation_type || raw.type || "unknown",
        description: raw.description,
        ...raw,
    };
    normalized.id = String(normalized.id);
    normalized.source = String(normalized.source || "");
    normalized.target = String(normalized.target || "");
    normalized.relation_type = normalized.relation_type || normalized.type || "unknown";
    return { data: normalized };
}

function toGraphElements(payload) {
    if (payload.elements) {
        const nodes = payload.elements.nodes || [];
        const edges = payload.elements.edges || [];
        return [
            ...nodes.map((node, index) => normalizeGraphNode(node, index)),
            ...edges.map((edge, index) => normalizeGraphEdge(edge, index)),
        ];
    }
    if (Array.isArray(payload)) {
        return payload.map((item, index) => {
            const data = item?.data || item || {};
            if (data.source !== undefined && data.target !== undefined) {
                return normalizeGraphEdge(item, index);
            }
            return normalizeGraphNode(item, index);
        });
    }
    const nodes = (payload.nodes || []).map((node, index) => normalizeGraphNode(node, index));
    const edges = (payload.links || payload.edges || []).map((edge, index) => normalizeGraphEdge(edge, index));
    return [...nodes, ...edges];
}

async function refreshSystemStatus() {
    try {
        const [statusResp, healthResp] = await Promise.all([
            fetch(`${API_BASE}/system-status`),
            fetch(`${API_BASE}/system/health-check${selectedDocId ? `?document_id=${encodeURIComponent(selectedDocId)}` : ""}`),
        ]);

        if (statusResp.ok) {
            const statusPayload = await statusResp.json();
            const llmAvailable = statusPayload.services?.llm?.status === "available";
            const llmPill = document.getElementById("llm-pill");
            llmPill.textContent = `LLM: ${llmAvailable ? "connected" : "degraded"}`;
            llmPill.style.background = llmAvailable ? "rgba(71, 208, 125, 0.25)" : "rgba(232, 179, 75, 0.25)";
        }

        if (healthResp.ok) {
            const healthPayload = await healthResp.json();
            const healthPill = document.getElementById("health-pill");
            healthPill.textContent = `System: ${healthPayload.status}`;
            healthPill.style.background =
                healthPayload.status === "healthy"
                    ? "rgba(71, 208, 125, 0.25)"
                    : "rgba(232, 179, 75, 0.25)";
        }
    } catch (error) {
        // Keep UI responsive when backend is down.
    }
}

function isDocumentAnalyzed(doc) {
    return String(doc?.status || "").toLowerCase() === "analyzed";
}

function renderDatasetStats(stats) {
    const validationRate = Number(stats.validationRate || 0);
    const datasetFolder = stats.exportFolder || "unknown";
    const latestFile = stats.latestExport || "Will appear after export";
    document.getElementById("dataset-stats").innerHTML = `
        Records: ${stats.total_records || 0}<br>
        Labeled: ${stats.labeled_records || 0}<br>
        Unlabeled: ${stats.unlabeled_records || 0}<br>
        Validation Rate: ${(validationRate * 100).toFixed(1)}%<br>
        Dataset Folder: ${escapeHtml(datasetFolder)}<br>
        Latest Export: ${escapeHtml(latestFile)}
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
            renderDatasetStats({
                total_records: 0,
                labeled_records: 0,
                unlabeled_records: 0,
                exportFolder: "Awaiting report",
                latestExport: "No export yet",
            });
            return;
        }
        const payload = await response.json();
        renderAnalysisReport(payload);
        renderDatasetStatusFromReport(payload);
    } catch (error) {
        reportBox.textContent = "Failed to load analysis report.";
    }
}

function renderDatasetStatusFromReport(report) {
    const artifacts = report.artifacts || {};
    const exportPath = normalizeDatasetExportPath(artifacts.dataset_export);
    renderDatasetStats({
        total_records: Number(artifacts.dataset_records || 0),
        labeled_records: 0,
        unlabeled_records: Number(artifacts.dataset_records || 0),
        validationRate: 0,
        exportFolder: exportPath ? exportPath.replace(/[\\/][^\\/]+$/, "") : "D:\\Clarion\\data\\datasets",
        latestExport: exportPath || "No export yet",
    });
}

function normalizeDatasetExportPath(pathValue) {
    const raw = String(pathValue || "").trim();
    if (!raw) return "";
    if (/^[a-zA-Z]:\\/.test(raw)) return raw;
    const fileName = raw.split(/[\\/]/).pop();
    return fileName ? `D:\\Clarion\\data\\datasets\\${fileName}` : raw;
}

function renderAnalysisReport(report) {
    const reportBox = document.getElementById("analysis-report");
    const insights = report.insights || {};
    const validationFailures = (report.stage_validations || []).filter(
        (item) => item.validation_passed === false
    );
    const failedStages = validationFailures
        .map((item) => `${item.stage}: ${(item.issues || []).join(", ") || "validation failed"}`)
        .join(" | ");
    const topConcepts = (insights.top_concepts || [])
        .map((item) => `${item.name} (${item.chunk_mentions})`)
        .join(", ");
    const relationBreakdown = (insights.relation_breakdown || [])
        .map((item) => `${item.type}: ${item.count}`)
        .join(", ");
    const sampleRelations = (insights.sample_relations || [])
        .map(
            (item) =>
                `${escapeHtml(item.from)} -[${escapeHtml(item.type)}]-> ${escapeHtml(item.to)} (${escapeHtml(item.confidence)})`
        )
        .join("<br>");
    const findings = (insights.key_findings || [])
        .map((item) => `- ${escapeHtml(item)}`)
        .join("<br>");
    const narrativeSummary = insights.narrative_summary || insights.overview || "No summary available.";

    reportBox.innerHTML = `
        <div class="report-header">
            <span class="report-tag ${report.success ? 'success' : (report.status === 'uploaded' || report.status === 'processing' ? 'pending' : 'error')}">
                ${report.success ? "Success" : (report.status === 'uploaded' || report.status === 'processing' ? "Pending" : "Failed")}
            </span>
            <span class="report-date">${report.generated_at ? new Date(report.generated_at).toLocaleString() : "n/a"}</span>
        </div>
        
        <div class="report-section">
            <h3>Summary Overview</h3>
            <div class="narrative-summary">${escapeHtml(narrativeSummary)}</div>
        </div>

        <!-- Graph metrics (Graph Complexity, Extraction Quality, Risk Assessment) removed to cleaner UI. See hidden_metrics_reference.md for details. -->

        <div class="report-section">
            <h3>Key Findings</h3>
            <div class="findings-list">${findings || "No key findings identified."}</div>
        </div>

        <div class="report-footer">
            <strong>Stages:</strong> ${(report.stages_completed || []).join(", ") || "none"}
            ${failedStages ? `<div class="validation-error"><strong>Issues:</strong> ${escapeHtml(failedStages)}</div>` : ''}
        </div>
    `;
}


function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function setStatus(elementId, message, type = "") {
    const element = document.getElementById(elementId);
    element.textContent = message;
    element.className = `status-box ${type}`.trim();
}
