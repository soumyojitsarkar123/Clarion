const API_BASE = 'http://localhost:8000';

let cy = null;
let selectedDocId = null;
let selectedDocName = null;

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

function initializeApp() {
    // Setup file upload
    const fileInput = document.getElementById('file-input');
    const dropzone = document.querySelector('.dropzone');
    const uploadForm = document.getElementById('upload-form');

    fileInput.addEventListener('change', handleFileSelect);
    
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(event => {
        dropzone.addEventListener(event, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(event => {
        dropzone.addEventListener(event, () => dropzone.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(event => {
        dropzone.addEventListener(event, () => dropzone.classList.remove('dragover'), false);
    });

    dropzone.addEventListener('drop', handleDrop, false);
    dropzone.addEventListener('click', () => fileInput.click());
    uploadForm.addEventListener('submit', handleUpload);

    // Setup buttons
    document.getElementById('refresh-btn').addEventListener('click', loadDocuments);
    document.getElementById('analyze-btn').addEventListener('click', handleAnalyze);
    document.getElementById('zoom-in').addEventListener('click', () => cy && cy.zoom(cy.zoom() * 1.2));
    document.getElementById('zoom-out').addEventListener('click', () => cy && cy.zoom(cy.zoom() * 0.8));
    document.getElementById('fit-graph').addEventListener('click', () => cy && cy.fit());

    // Initialize graph
    initGraph();

    // Load data
    loadDocuments();
    checkLLMStatus();
    setInterval(checkLLMStatus, 3000);
}

// File handling
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length) {
        handleFiles(files);
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    document.getElementById('file-input').files = files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length) {
        const file = files[0];
        const isValid = file.type === 'application/pdf' || file.type === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
        if (!isValid) {
            showStatus('upload-status', 'Please select a PDF or DOCX file', 'error');
            return;
        }
        if (file.size > 50 * 1024 * 1024) {
            showStatus('upload-status', 'File size exceeds 50MB limit', 'error');
            return;
        }
    }
}

// Upload handler
async function handleUpload(e) {
    e.preventDefault();
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];

    if (!file) {
        showStatus('upload-status', 'Please select a file', 'error');
        return;
    }

    showStatus('upload-status', 'Uploading...', 'info');

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        showStatus('upload-status', 'Upload successful!', 'success');
        fileInput.value = '';
        setTimeout(() => loadDocuments(), 1000);

    } catch (error) {
        showStatus('upload-status', `Error: ${error.message}`, 'error');
    }
}

// Load documents
async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/upload/list`);
        const data = await response.json();
        const documents = data.documents || [];

        const listContainer = document.getElementById('documents-list');
        
        if (documents.length === 0) {
            listContainer.innerHTML = '<p class="empty-state">No documents</p>';
            updateStats(documents);
            return;
        }

        listContainer.innerHTML = '';
        documents.forEach(doc => {
            const item = document.createElement('div');
            item.className = 'document-item';
            if (doc.id === selectedDocId) item.classList.add('selected');

            const statusEmoji = doc.status === 'completed' ? '✅' : 
                              doc.status === 'processing' ? '⏳' : '📄';

            item.innerHTML = `
                <div class="document-item-title">${statusEmoji} ${doc.name}</div>
                <div class="document-item-meta">${(doc.size / 1024 / 1024).toFixed(1)}MB • ${doc.status}</div>
            `;

            item.addEventListener('click', () => selectDocument(doc, item));
            listContainer.appendChild(item);
        });

        updateStats(documents);

    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

// Select document
function selectDocument(doc, element) {
    selectedDocId = doc.id;
    selectedDocName = doc.name;

    document.querySelectorAll('.document-item').forEach(el => el.classList.remove('selected'));
    element.classList.add('selected');

    const infoBox = document.getElementById('doc-info');
    const analyzeBtn = document.getElementById('analyze-btn');

    infoBox.innerHTML = `
        <strong>${doc.name}</strong><br>
        <span style="color: #64748b; font-size: 12px;">Status: ${doc.status} | Size: ${(doc.size / 1024 / 1024).toFixed(1)}MB</span>
    `;

    if (doc.status !== 'completed' && doc.status !== 'processing') {
        analyzeBtn.style.display = 'flex';
    } else {
        analyzeBtn.style.display = 'none';
    }

    loadGraphData(doc.id);
}

// Analyze document
async function handleAnalyze() {
    if (!selectedDocId) {
        showStatus('analysis-status', 'Select a document first', 'error');
        return;
    }

    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.textContent = 'Processing...';

    try {
        showStatus('analysis-status', 'Starting analysis...', 'info');

        const response = await fetch(`${API_BASE}/analyze/${selectedDocId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                generate_hierarchy: true,
                run_evaluation: false
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Analysis failed');
        }

        showStatus('analysis-status', `Analysis started! Job: ${data.job_id}`, 'success');

        // Poll for completion
        let attempts = 0;
        const pollInterval = setInterval(async () => {
            attempts++;
            if (attempts > 120) {
                clearInterval(pollInterval);
                btn.disabled = false;
                btn.textContent = 'Start Analysis';
                return;
            }

            try {
                const statusResp = await fetch(`${API_BASE}/status/${selectedDocId}`);
                const statusData = await statusResp.json();

                if (statusData.overall_status === 'completed') {
                    clearInterval(pollInterval);
                    showStatus('analysis-status', 'Analysis completed!', 'success');
                    btn.disabled = true;
                    btn.style.display = 'none';
                    loadDocuments();
                    loadGraphData(selectedDocId);
                    loadResults(selectedDocId);
                }
            } catch (err) {
                console.error('Poll error:', err);
            }
        }, 2000);

    } catch (error) {
        showStatus('analysis-status', `Error: ${error.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Start Analysis';
    }
}

// Load and display results
async function loadResults(docId) {
    try {
        const kmResponse = await fetch(`${API_BASE}/knowledge-map/${docId}`);
        const kmData = await kmResponse.json();

        const resultsPanel = document.getElementById('results-panel');
        resultsPanel.classList.add('show');

        // Display concepts
        if (kmData.concepts) {
            const conceptsList = document.getElementById('concepts-list');
            conceptsList.innerHTML = '';
            kmData.concepts.slice(0, 5).forEach(concept => {
                const span = document.createElement('span');
                span.textContent = concept.name;
                conceptsList.appendChild(span);
            });
        }

        // Display summary
        try {
            const summaryResponse = await fetch(`${API_BASE}/summary/${docId}`);
            const summaryData = await summaryResponse.json();
            if (summaryData.summary) {
                document.getElementById('summary-text').textContent = summaryData.summary.slice(0, 200) + '...';
            }
        } catch (err) {
            console.log('Summary not available yet');
        }

    } catch (error) {
        console.error('Error loading results:', error);
    }
}

// Graph
function initGraph() {
    const container = document.getElementById('graph-container');
    cy = cytoscape({
        container: container,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#3b82f6',
                    'label': 'data(label)',
                    'color': '#1e293b',
                    'font-size': '11px',
                    'text-valign': 'bottom',
                    'text-margin-y': 4,
                    'width': 30,
                    'height': 30
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 1.5,
                    'line-color': '#cbd5e1',
                    'target-arrow-color': '#cbd5e1',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier'
                }
            }
        ],
        layout: { name: 'grid' }
    });
}

async function loadGraphData(docId) {
    try {
        const kmResponse = await fetch(`${API_BASE}/knowledge-map/${docId}`);
        const kmData = await kmResponse.json();

        const elements = [];
        const nodeIds = new Set();

        if (kmData.concepts) {
            kmData.concepts.forEach(concept => {
                nodeIds.add(concept.id);
                elements.push({
                    data: { id: concept.id, label: concept.name }
                });
            });
        }

        if (kmData.relations) {
            kmData.relations.forEach((rel, idx) => {
                if (nodeIds.has(rel.from_concept) && nodeIds.has(rel.to_concept)) {
                    elements.push({
                        data: {
                            id: `edge-${idx}`,
                            source: rel.from_concept,
                            target: rel.to_concept
                        }
                    });
                }
            });
        }

        if (cy) {
            cy.elements().remove();
            cy.add(elements);
            cy.layout({ name: 'dagre', rankDir: 'LR', spacingFactor: 1.5 }).run();
            document.getElementById('graph-info').textContent = `${kmData.concepts?.length || 0} concepts, ${kmData.relations?.length || 0} relations`;
        }

    } catch (error) {
        console.error('Graph error:', error);
        document.getElementById('graph-info').textContent = 'Error loading graph';
    }
}

// LLM Status
async function checkLLMStatus() {
    try {
        const response = await fetch(`${API_BASE}/system-status`);
        const data = await response.json();

        const indicator = document.getElementById('llm-indicator');
        const text = document.getElementById('llm-text');

        if (data.services.llm.status === 'available') {
            indicator.classList.add('active');
            text.textContent = 'LLM: Connected';
        } else {
            indicator.classList.remove('active');
            text.textContent = 'LLM: Demo Mode';
        }

        document.getElementById('stat-memory').textContent = Math.round(data.system.memory_mb) + 'MB';

    } catch (error) {
        document.getElementById('llm-indicator').classList.remove('active');
        document.getElementById('llm-text').textContent = 'LLM: Error';
    }
}

// Update stats
function updateStats(documents) {
    const total = documents.length;
    const processing = documents.filter(d => d.status === 'processing').length;
    const completed = documents.filter(d => d.status === 'completed').length;

    document.getElementById('stat-total').textContent = total;
    document.getElementById('stat-processing').textContent = processing;
    document.getElementById('stat-completed').textContent = completed;
}

// Status message
function showStatus(elementId, message, type) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = `status-box show ${type}`;

    if (type !== 'error') {
        setTimeout(() => {
            el.classList.remove('show');
        }, 4000);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initCytoscape();
    initEventListeners();
    loadDocuments();
    pollSystemStatus();
    setInterval(() => pollSystemStatus(), 3000);
});

function initEventListeners() {
    // File upload
    const fileInput = document.getElementById('file-input');
    const fileInputBox = document.querySelector('.file-input-box');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            document.getElementById('file-name').textContent = '✓ ' + fileInput.files[0].name;
            document.getElementById('file-name').style.display = 'block';
        }
    });

    fileInputBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        fileInputBox.classList.add('dragover');
    });

    fileInputBox.addEventListener('dragleave', () => {
        fileInputBox.classList.remove('dragover');
    });

    fileInputBox.addEventListener('drop', (e) => {
        e.preventDefault();
        fileInputBox.classList.remove('dragover');
        fileInput.files = e.dataTransfer.files;
        if (fileInput.files.length > 0) {
            document.getElementById('file-name').textContent = '✓ ' + fileInput.files[0].name;
            document.getElementById('file-name').style.display = 'block';
        }
    });

    document.getElementById('upload-form').addEventListener('submit', handleUpload);
    document.getElementById('refresh-docs').addEventListener('click', loadDocuments);
    document.getElementById('analyze-btn').addEventListener('click', handleAnalyze);

    // Graph controls
    document.getElementById('zoom-in').addEventListener('click', () => {
        if (cy) cy.zoom(cy.zoom() * 1.2);
    });

    document.getElementById('zoom-out').addEventListener('click', () => {
        if (cy) cy.zoom(cy.zoom() * 0.8);
    });

    document.getElementById('fit-graph').addEventListener('click', () => {
        if (cy && cy.elements().length > 0) cy.fit();
    });

    document.getElementById('layout-hier').addEventListener('click', () => {
        if (cy && cy.elements().length > 0) {
            cy.layout({ name: 'dagre', rankDir: 'TB' }).run();
        }
    });
}

async function handleUpload(e) {
    e.preventDefault();

    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];

    if (!file) {
        showStatus('upload-status', '❌ Please select a file', 'error');
        return;
    }

    try {
        showStatus('upload-status', '⏳ Uploading...', 'info');

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || 'Upload failed');
        }

        showStatus('upload-status', '✅ Uploaded successfully!', 'success');

        fileInput.value = '';
        document.getElementById('file-name').style.display = 'none';

        setTimeout(() => {
            loadDocuments();
            showStatus('upload-status', '', '');
        }, 1000);

    } catch (error) {
        showStatus('upload-status', `❌ ${error.message}`, 'error');
    }
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/upload/list`);
        if (!response.ok) throw new Error('Failed to load documents');

        const data = await response.json();
        const documents = data.documents || [];

        const listContainer = document.getElementById('documents-list');

        if (documents.length === 0) {
            listContainer.innerHTML = '<p class="empty-state">📭 No documents uploaded yet</p>';
            updateStats(documents);
            return;
        }

        listContainer.innerHTML = '';
        documents.forEach(doc => {
            const item = createDocumentItem(doc);
            listContainer.appendChild(item);
        });

        updateStats(documents);

    } catch (error) {
    }
}

function createDocumentItem(doc) {
    const item = document.createElement('div');
    item.className = 'document-item';
    if (doc.id === selectedDocId) item.classList.add('selected');

    const statusEmoji = doc.status === 'completed' ? '✅' : 
                        doc.status === 'processing' ? '⏳' : 
                        doc.status === 'failed' ? '❌' : '📄';

    const size = (doc.size / 1024 / 1024).toFixed(2);

    item.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 4px;">${statusEmoji} ${doc.name}</div>
        <div style="font-size: 0.85rem; color: var(--text-secondary);">
            ${size}MB • ${doc.status.toUpperCase()}
        </div>
    `;

    item.addEventListener('click', () => selectDocument(doc, item));
    return item;
}

function selectDocument(doc, element) {
    selectedDocId = doc.id;
    selectedDocName = doc.name;

    document.querySelectorAll('.document-item').forEach(item => item.classList.remove('selected'));
    element.classList.add('selected');

    const infoBox = document.getElementById('current-doc-info');
    const analyzeBtn = document.getElementById('analyze-btn');

    infoBox.innerHTML = `
        <strong>Document:</strong> ${doc.name}<br>
        <strong>Status:</strong> ${doc.status.toUpperCase()}<br>
        <strong>Size:</strong> ${(doc.size / 1024 / 1024).toFixed(2)}MB
    `;

    if (doc.status !== 'completed' && doc.status !== 'processing') {
        analyzeBtn.style.display = 'flex';
    } else {
        analyzeBtn.style.display = 'none';
    }

    loadGraph(doc.id);
}

async function handleAnalyze() {
    if (!selectedDocId) {
        showStatus('analysis-status', '❌ Select a document first', 'error');
        return;
    }

    const btn = document.getElementById('analyze-btn');
    btn.disabled = true;
    btn.textContent = 'Starting...';

    try {
        const response = await fetch(`${API_BASE}/analyze/${selectedDocId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (response.ok) {
            if (data.status === 'already_analyzed') {
                showStatus('analysis-status', 'Document already analyzed', 'success');
                btn.disabled = true;
                btn.textContent = 'Already Analyzed';
            } else {
                showStatus('analysis-status', `Analysis started (Job: ${data.job_id})`, 'info');
                btn.textContent = 'Processing...';
                startPolling(selectedDocId);
            }
        } else {
            showStatus('analysis-status', `Error: ${data.detail || 'Analysis failed'}`, 'error');
            btn.disabled = false;
            btn.textContent = 'Start Analysis';
        }
    } catch (error) {
        showStatus('analysis-status', `Error: ${error.message}`, 'error');
        btn.disabled = false;
        btn.textContent = 'Start Analysis';
    }
}

async function loadGraph(docId) {
    const graphPanel = document.getElementById('graph-panel');
    const graphInfo = document.getElementById('graph-info');
    
    try {
        const response = await fetch(`${API_BASE}/graph/${docId}`);
        
        if (!response.ok) {
            const kmResponse = await fetch(`${API_BASE}/knowledge-map/${docId}`);
            if (!kmResponse.ok) {
                graphInfo.textContent = 'No graph available. Please analyze the document first.';
                if (cy) {
                    cy.elements().remove();
                }
                return;
            }
            const kmData = await kmResponse.json();
            renderGraphFromKnowledgeMap(kmData);
            return;
        }

        const data = await response.json();
        
        if (!data.nodes || data.nodes.length === 0) {
            const kmResponse = await fetch(`${API_BASE}/knowledge-map/${docId}`);
            if (kmResponse.ok) {
                const kmData = await kmResponse.json();
                renderGraphFromKnowledgeMap(kmData);
                return;
            }
            graphInfo.textContent = 'No graph data available.';
            return;
        }

        renderCytoscapeGraph(data);
    } catch (error) {
        console.error('Graph load error:', error);
        
        try {
            const kmResponse = await fetch(`${API_BASE}/knowledge-map/${docId}`);
            if (kmResponse.ok) {
                const kmData = await kmResponse.json();
                renderGraphFromKnowledgeMap(kmData);
                return;
            }
        } catch (e) {}
        
        graphInfo.textContent = `Error loading graph: ${error.message}`;
    }
}

function renderGraphFromKnowledgeMap(kmData) {
    const elements = [];
    const nodeIds = new Set();

    if (kmData.concepts) {
        kmData.concepts.forEach(concept => {
            nodeIds.add(concept.id);
            elements.push({
                data: {
                    id: concept.id,
                    label: concept.name,
                    type: 'concept'
                }
            });
        });
    }

    if (kmData.relations) {
        kmData.relations.forEach((rel, idx) => {
            if (nodeIds.has(rel.from_concept) && nodeIds.has(rel.to_concept)) {
                elements.push({
                    data: {
                        id: `edge-${idx}`,
                        source: rel.from_concept,
                        target: rel.to_concept,
                        label: rel.relation_type,
                        weight: rel.confidence
                    }
                });
            }
        });
    }

    document.getElementById('graph-info').textContent = 
        `Graph: ${kmData.concepts?.length || 0} concepts, ${kmData.relations?.length || 0} relations`;

    renderCytoscapeGraph({ elements });
}

function renderCytoscapeGraph(graphData) {
    const container = document.getElementById('cy');
    
    if (cy) {
        cy.destroy();
    }

    const elements = graphData.elements || graphData;
    
    if (!elements || elements.length === 0) {
        document.getElementById('graph-info').textContent = 'No graph elements to display';
        return;
    }

    cy = cytoscape({
        container: container,
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#3498db',
                    'label': 'data(label)',
                    'color': '#2c3e50',
                    'font-size': '12px',
                    'text-valign': 'bottom',
                    'text-margin-y': 5,
                    'width': 40,
                    'height': 40
                }
            },
            {
                selector: 'node[type="concept"]',
                style: {
                    'background-color': '#3498db',
                    'shape': 'ellipse'
                }
            },
            {
                selector: 'node[type="topic"]',
                style: {
                    'background-color': '#e74c3c',
                    'shape': 'rectangle'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#95a5a6',
                    'target-arrow-color': '#95a5a6',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '10px',
                    'color': '#7f8c8d'
                }
            },
            {
                selector: ':selected',
                style: {
                    'background-color': '#2ecc71',
                    'line-color': '#2ecc71',
                    'target-arrow-color': '#2ecc71'
                }
            }
        ],
        layout: {
            name: 'dagre',
            rankDir: 'LR',
            nodeSep: 50,
            rankSep: 100
        }
    });

    cy.on('tap', 'node', function(evt){
        const node = evt.target;
        const info = document.getElementById('graph-info');
        info.textContent = `Selected: ${node.data('label')} (${node.id()})`;
    });

    document.getElementById('graph-info').textContent = 
        `Graph: ${cy.nodes().length} nodes, ${cy.edges().length} edges`;
}

function applyLayout(layoutName) {
    if (!cy) return;
    
    cy.layout({
        name: layoutName,
        rankDir: 'LR',
        nodeSep: 50,
        rankSep: 100,
        animate: true,
        animationDuration: 500
    }).run();
}

function showStatus(elementId, message, type) {
    const el = document.getElementById(elementId);
    el.textContent = message;
    el.className = `status-message show ${type}`;
    
    if (type === 'success' || type === 'info') {
        setTimeout(() => {
            el.classList.remove('show');
        }, 5000);
    }
}
