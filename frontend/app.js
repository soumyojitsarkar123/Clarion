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
        const filterVal = e.target.value;
        if (graph) {
            graph.edges().forEach(edge => {
                if (filterVal === "all" || filterVal === "none" || edge.data('relation_type') === filterVal) {
                    edge.style('display', 'element');
                } else {
                    edge.style('display', 'none');
                }
            });
        }
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
    loadGraph(doc.id);
    refreshInspectorPanels();
    refreshPipelineStatus();
    loadAnalysisReport();
    monitorPipeline();
}

async function startAnalysis() {
    if (!selectedDocId) return;
    const analyzeBtn = document.getElementById("analyze-btn");
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Submitting...";
    setStatus("analysis-status", "Submitting pipeline job...", "info");
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
        await refreshInspectorPanels();
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
                        loadDocuments();
                        loadGraph(selectedDocId);
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
            loadDocuments();
            loadGraph(selectedDocId);
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
    const timestamp = payload.timestamp || new Date().toISOString();
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
    lineSpan.textContent = `[${timestamp}] [${stage}] [${event}] ${message}\n`;
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
                        if (type === 'topic') return '#38bdf8';
                        if (type === 'subtopic') return '#818cf8';
                        if (type === 'document') return '#f43f5e';
                        return '#1f7c70';
                    },
                    shape: function (ele) {
                        const type = ele.data('type');
                        if (type === 'topic') return 'hexagon';
                        if (type === 'subtopic') return 'diamond';
                        if (type === 'document') return 'star';
                        return 'ellipse';
                    },
                    label: "data(label)",
                    color: "#f8fafc",
                    "text-valign": "bottom",
                    "text-halign": "center",
                    "text-margin-y": 4,
                    "text-outline-color": "#0f172a",
                    "text-outline-width": 2,
                    "font-size": function (ele) {
                        const type = ele.data('type');
                        if (type === 'topic') return '13px';
                        if (type === 'subtopic') return '11px';
                        return '9px';
                    },
                    "font-weight": function (ele) {
                        return (ele.data('type') === 'topic' || ele.data('type') === 'subtopic') ? 'bold' : 'normal';
                    },
                    width: function (ele) {
                        const type = ele.data('type');
                        if (type === 'topic') return 50;
                        if (type === 'subtopic') return 40;
                        if (type === 'document') return 60;
                        return 30;
                    },
                    height: function (ele) {
                        const type = ele.data('type');
                        if (type === 'topic') return 50;
                        if (type === 'subtopic') return 40;
                        if (type === 'document') return 60;
                        return 30;
                    },
                    "border-width": 2,
                    "border-color": "#475569",
                    "text-outline-width": 1,
                    "text-outline-color": "#0b1120"
                },
            },
            {
                selector: "edge",
                style: {
                    width: 1.5,
                    "line-color": "#475569",
                    "target-arrow-color": "#475569",
                    "target-arrow-shape": "triangle",
                    "curve-style": "taxi",
                    "taxi-direction": "downward",
                    "edge-distances": "node-position",
                    opacity: 0.6,
                    label: "data(relation_type)",
                    "font-size": "8px",
                    "text-rotation": "autorotate",
                    "text-margin-y": -10,
                    "color": "#94a3b8"
                },
            },
        ],
        layout: {
            name: "breadthfirst",
            directed: true,
            padding: 50,
            spacingFactor: 1.25,
            animate: true,
            nodeDimensionsIncludeLabels: true
        },
        autoungrabify: true,
        userZoomingEnabled: true,
        userPanningEnabled: true,
        boxSelectionEnabled: false
    });

    graph.on('mouseover', 'node', function (e) {
        const data = e.target.data();
        const tooltip = document.getElementById("graph-tooltip");
        const centrality = data.centrality !== undefined ? Number(data.centrality).toFixed(3) : "n/a";
        const community = data.community || "n/a";
        const definition = data.definition || data.description || "No definition available.";
        tooltip.innerHTML = `<strong>${escapeHtml(data.label)}</strong><br>
                             <div>Community: ${escapeHtml(community)}</div>
                             <div>Centrality: ${escapeHtml(centrality)}</div>
                             <div style="font-size: 0.75rem; margin-top: 6px; color: #cbd5e1">${escapeHtml(definition)}</div>`;
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

    graph.on('mouseout', 'node', function () {
        document.getElementById("graph-tooltip").style.display = "none";
    });
}

async function loadGraph(documentId) {
    try {
        const response = await fetch(`${API_BASE}/graph/${documentId}`);
        if (!response.ok) {
            document.getElementById("graph-info").textContent = "No graph available.";
            graph.elements().remove();
            return;
        }
        const payload = await response.json();
        const elements = toGraphElements(payload);
        graph.elements().remove();
        graph.add(elements);
        graph.layout({
            name: "breadthfirst",
            directed: true,
            spacingFactor: 2.0,
            roots: cy => cy.nodes('[type = "document"]'),
            animate: true
        }).run();
        graph.autoungrabify(true);
        document.getElementById("graph-info").textContent = `${graph.nodes().length} nodes / ${graph.edges().length} edges`;
        populateGraphFilterAndTable(elements);
    } catch (error) {
        document.getElementById("graph-info").textContent = "Graph load failed.";
    }
}

function populateGraphFilterAndTable(elementsArr) {
    const nodes = elementsArr.filter(e => e.data && (e.data.source === undefined));
    const edges = elementsArr.filter(e => e.data && e.data.source !== undefined);

    const tbody = document.querySelector("#concepts-table tbody");
    tbody.innerHTML = "";

    let sortedNodes = [...nodes].sort((a, b) => (Number(b.data?.centrality || 0) - Number(a.data?.centrality || 0)));
    if (sortedNodes.length === 0) {
        tbody.innerHTML = `<tr><td colspan="4" class="muted" style="text-align: center;">No concepts loaded</td></tr>`;
    } else {
        const html = sortedNodes.slice(0, 100).map(n => {
            const data = n.data;
            const label = data.label || data.id;
            const desc = data.definition || data.description || "-";
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
    filter.innerHTML = `<option value="all">All Relations</option>`;
    edgeTypes.forEach(t => {
        const opt = document.createElement("option");
        opt.value = t;
        opt.textContent = t;
        filter.appendChild(opt);
    });
}

function toGraphElements(payload) {
    if (payload.elements) {
        const nodes = payload.elements.nodes || [];
        const edges = payload.elements.edges || [];
        return [...nodes, ...edges];
    }
    if (Array.isArray(payload)) {
        return payload;
    }
    const nodes = (payload.nodes || []).map((node) => ({
        data: {
            id: String(node.id),
            label: node.label || node.name || String(node.id),
            centrality: node.centrality,
            community: node.community,
            definition: node.definition || node.description,
            ...node
        },
    }));
    const edges = (payload.links || payload.edges || []).map((edge, index) => ({
        data: {
            id: String(edge.id || `e-${index}`),
            source: String(edge.source),
            target: String(edge.target),
            relation_type: edge.relation_type || edge.type || "unknown",
            ...edge
        },
    }));
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
            renderHealthStats(healthPayload);
        }
    } catch (error) {
        // Keep UI responsive when backend is down.
    }
}

async function refreshInspectorPanels() {
    if (!selectedDocId) {
        return;
    }
    try {
        const [graphResp, datasetResp, healthResp] = await Promise.all([
            fetch(`${API_BASE}/system/graph-stats?document_id=${encodeURIComponent(selectedDocId)}`),
            fetch(`${API_BASE}/system/dataset-stats?document_id=${encodeURIComponent(selectedDocId)}`),
            fetch(`${API_BASE}/system/health-check?document_id=${encodeURIComponent(selectedDocId)}`),
        ]);

        if (graphResp.ok) {
            const graphStats = await graphResp.json();
            renderGraphStats(graphStats);
        } else if (graphResp.status === 404) {
            await refreshLegacyGraphStats();
        }
        if (datasetResp.ok) {
            const datasetStats = await datasetResp.json();
            renderDatasetStats(datasetStats);
        } else if (datasetResp.status === 404) {
            await refreshLegacyDatasetStats();
        }
        if (healthResp.ok) {
            const healthStats = await healthResp.json();
            renderHealthStats(healthStats);
        } else if (healthResp.status === 404) {
            await refreshLegacyHealthStats();
        }
    } catch (error) {
        // Ignore periodic inspector errors.
    }
}

async function refreshLegacyGraphStats() {
    try {
        const response = await fetch(`${API_BASE}/knowledge-map/${encodeURIComponent(selectedDocId)}`);
        if (!response.ok) return;
        const payload = await response.json();
        renderGraphStats({
            concept_count: (payload.concepts || []).length,
            relation_count: (payload.relations || []).length,
            graph_density: "n/a (legacy)",
            central_nodes: [],
        });
    } catch (error) {
        // Ignore fallback errors.
    }
}

async function refreshLegacyDatasetStats() {
    try {
        const response = await fetch(`${API_BASE}/dataset/relations/stats`);
        if (!response.ok) return;
        const payload = await response.json();
        renderDatasetStats(payload);
    } catch (error) {
        // Ignore fallback errors.
    }
}

async function refreshLegacyHealthStats() {
    try {
        const response = await fetch(`${API_BASE}/system-status`);
        if (!response.ok) return;
        const payload = await response.json();
        renderHealthStats({
            status: "legacy",
            checks: {
                embedding_service_health: { status: payload.services?.embeddings || "unknown" },
                vectorstore_integrity: { status: "unknown" },
                graph_integrity: { status: payload.services?.graph_engine || "unknown" },
                dataset_integrity: { status: "unknown" },
                llm_connectivity: { status: payload.services?.llm?.status || "unknown" },
            },
        });
    } catch (error) {
        // Ignore fallback errors.
    }
}

function renderGraphStats(stats) {
    const centralNodes = (stats.central_nodes || [])
        .map((node) => `${node.label || node.node_id} (${Number(node.score || 0).toFixed(2)})`)
        .join(", ") || "None";
    document.getElementById("graph-stats").innerHTML = `
        Concepts: ${stats.concept_count || 0}<br>
        Relations: ${stats.relation_count || 0}<br>
        Density: ${stats.graph_density || 0}<br>
        Central Nodes: ${centralNodes}
    `;
}

function renderDatasetStats(stats) {
    const validation = stats.validation_stats || {};
    const datasetDb = stats.storage_paths?.relation_dataset_db || "unknown";
    document.getElementById("dataset-stats").innerHTML = `
        Records: ${stats.total_records || 0}<br>
        Labeled: ${stats.labeled_records || 0}<br>
        Unlabeled: ${stats.unlabeled_records || 0}<br>
        Validation Rate: ${(Number(validation.validation_rate || 0) * 100).toFixed(1)}%<br>
        Dataset DB: ${escapeHtml(datasetDb)}
    `;
}

function renderHealthStats(stats) {
    const checks = stats.checks || {};
    document.getElementById("system-health").innerHTML = `
        Status: ${stats.status || "unknown"}<br>
        Embedding: ${checks.embedding_service_health?.status || "unknown"}<br>
        Vectorstore: ${checks.vectorstore_integrity?.status || "unknown"}<br>
        Graph: ${checks.graph_integrity?.status || "unknown"}<br>
        Dataset: ${checks.dataset_integrity?.status || "unknown"}<br>
        LLM: ${checks.llm_connectivity?.status || "unknown"}
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
    const summary = report.summary || {};
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
    const recommendations = (insights.recommendations || [])
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

        <div class="report-section">
            <h3>Insights & Recommendations</h3>
            <div class="findings-list">${recommendations || "No recommendations generated."}</div>
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
