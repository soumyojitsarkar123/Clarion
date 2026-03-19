# Clarion - Professional Document Analysis System

## 🚀 Quick Start

### Backend Setup
```bash
cd Clarion-Backend
pip install -r requirements.txt
python main.py
```

The backend will start on `http://localhost:8000`

### Frontend Setup
```bash
cd frontend
python -m http.server 8080
```

Then open `http://localhost:8080` in your browser.

---

## 📋 Architecture

### Frontend
- **Framework**: Vanilla JavaScript + HTML5 + CSS3
- **Port**: 8080
- **Features**:
  - File drag-drop upload (PDF/DOCX, max 50MB)
  - Document list with status indicators
  - Real-time analysis with polling
  - Knowledge graph visualization using Cytoscape.js
  - System status monitoring (LLM, Database)
  - Statistics dashboard

### Backend
- **Framework**: FastAPI
- **Port**: 8000
- **API Endpoints**:
  - `POST /upload` - Upload document
  - `GET /upload/list` - List documents
  - `POST /analyze/{doc_id}` - Start analysis
  - `GET /status/{doc_id}` - Check document status
  - `GET /knowledge-map/{doc_id}` - Get knowledge graph
  - `GET /summary/{doc_id}` - Get document summary
  - `GET /system-status` - Check system health
  - `GET /graph/{doc_id}` - Get graph data

---

## 🎨 UI Components

### Header
- Application title
- LLM connection status (red/green indicator)
- Database status indicator
- Memory usage

### Left Panel
1. **Upload Card** - Drag-drop file upload with validation
2. **Documents Card** - List of uploaded documents with status
3. **Stats Card** - Total docs, analyzing count, completed count, memory

### Right Panel
1. **Analysis Card** - Document info + analysis button + results display
2. **Knowledge Graph Card** - Graph visualization with zoom controls

---

## 🔄 Workflow

1. **Upload Document**
   - Drag file into upload area or click to select
   - Supports PDF and DOCX formats
   - Maximum 50MB file size

2. **Analyze Document**
   - Select document from list
   - Click "Start Analysis" button
   - System polls for completion every 2 seconds
   - Results display when complete

3. **View Results**
   - Key concepts extracted from document
   - Document summary
   - Knowledge graph showing concept relationships

4. **Monitor System**
   - LLM status (connected/demo mode)
   - System memory usage
   - Processing statistics

---

## ⚙️ API Integration

### File Upload
```javascript
const formData = new FormData();
formData.append('file', file);
await fetch('http://localhost:8000/upload', {
    method: 'POST',
    body: formData
});
```

### Start Analysis
```javascript
await fetch(`http://localhost:8000/analyze/${doc_id}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        generate_hierarchy: true,
        run_evaluation: false
    })
});
```

### Poll Status
```javascript
const response = await fetch(`http://localhost:8000/status/${doc_id}`);
const data = await response.json();
if (data.overall_status === 'completed') {
    // Analysis complete
}
```

---

## 🛠️ Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Graph Visualization**: Cytoscape.js
- **Backend**: Python FastAPI
- **LLM**: Ollama at http://localhost:11434/v1
- **Database**: SQLite
- **Document Processing**: PDF + DOCX support

---

## 🎯 Features

✅ Professional modern UI design
✅ Real-time document analysis
✅ Knowledge graph visualization
✅ Drag-drop file upload
✅ System health monitoring
✅ Responsive layout
✅ Real-time status polling
✅ Document management
✅ Statistics dashboard
✅ Error handling & status messages

---

## 📊 Professional Design

The UI features:
- Clean blue color scheme (#2563eb primary)
- Consistent typography using system fonts
- Card-based layout with subtle shadows
- Status indicators for real-time feedback
- Smooth transitions and hover effects
- Accessible color contrast
- Mobile-responsive design

---

## 🔍 Current Features

### Document Management
- Upload documents (PDF/DOCX)
- View document list with status
- Track document processing progress
- Delete documents (future)

### Analysis
- Automatic chunking
- Concept extraction
- Relationship mapping
- Hierarchy generation
- Summary generation

### Visualization
- Interactive knowledge graph
- Node and relationship visualization
- Zoom controls
- Auto-layout (Dagre)
- Graph statistics

### Monitoring
- LLM availability status
- System memory usage
- Processing queue status
- Job history
- Real-time metrics

---

## 🚀 Future Enhancements

- Export results (PDF, JSON)
- Advanced filtering and search
- Custom analysis parameters
- Batch processing
- User authentication
- Multi-language support
- Custom embedding models
- Advanced graph analysis

---

## 💡 Tips

1. **File Upload**: Large files may take longer. Use progress indicators for user feedback.
2. **Analysis**: Server-side chunking produces better results than client-side. Polling interval is set to 2 seconds.
3. **Graph Visualization**: Use zoom and fit buttons to navigate large graphs. Concepts are automatically laid out using Dagre algorithm.
4. **LLM Integration**: Ensure Ollama is running at `http://localhost:11434/v1` for full functionality.

---

Created with ❤️ for professional document analysis.
