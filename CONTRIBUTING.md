# Contributing to Clarion

Thank you for your interest in contributing to Clarion! This document provides guidelines and instructions for contributing.

## 🤝 Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Accept criticism gracefully
- Focus on the code, not the person

## 🔄 Development Workflow

### 1. Setup Development Environment

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/Clarion.git
cd Clarion

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Create .env file
cp Clarion-Backend/.env.example Clarion-Backend/.env
```

### 2. Create Feature Branch

```bash
# Always create a new branch from main
git checkout -b feature/your-feature-name

# Branch naming conventions:
# - feature/description - New feature
# - bugfix/description - Bug fix
# - docs/description - Documentation
# - refactor/description - Code refactoring
# - tests/description - Test additions
```

### 3. Make Changes

```bash
# Make your changes to the code
# Keep commits atomic and meaningful
git add .
git commit -m "feat: add new feature description"
```

### Commit Message Format

```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `tests`: Adding or updating tests
- `chore`: Build process, dependencies

**Example**:
```
feat: add document export to PDF

Add functionality to export analyzed documents as PDF reports
including knowledge graph and summary.

Closes #123
```

## 📝 Code Style Guide

### Python Code

```python
# Format with Black
black Clarion-Backend/

# Lint with Flake8
flake8 Clarion-Backend/ --max-line-length=100

# Type checking
mypy Clarion-Backend/
```

### Python Conventions

```python
# Use type hints
def analyze_document(doc_id: str, generate_hierarchy: bool = True) -> dict:
    """
    Analyze a document and extract knowledge.
    
    Args:
        doc_id: Document identifier
        generate_hierarchy: Whether to generate concept hierarchy
        
    Returns:
        Analysis results dictionary
        
    Raises:
        DocumentNotFound: If document doesn't exist
    """
    pass

# Use meaningful variable names
result = process_chunks(chunks)  # Good
r = pc(c)  # Bad

# Add docstrings to all functions
class KnowledgeMapService:
    """Service for building and managing knowledge maps."""
    pass

# Use constants for magic values
MAX_CHUNK_SIZE = 512
MIN_CHUNK_SIZE = 100
```

### JavaScript/Frontend Code

```javascript
// Use meaningful names
const documentsContainer = document.getElementById('documents-list');

// Use const by default, let when needed, avoid var
const fixedValue = 10;
let changingValue = 20;

// Use arrow functions when appropriate
const filterDocuments = (docs) => docs.filter(d => d.status === 'completed');

// Add comments for complex logic
// Poll for completion every 2 seconds until done or timeout
const pollInterval = setInterval(() => {
    checkStatus(docId).then(status => {
        if (status.overall_status === 'completed') {
            clearInterval(pollInterval);
        }
    });
}, 2000);
```

### CSS Conventions

```css
/* Use BEM naming: Block__Element--Modifier */
.card { }
.card__header { }
.card__header--active { }

/* Use CSS variables for colors */
:root {
    --primary: #2563eb;
    --success: #10b981;
}

.button {
    background: var(--primary);
}
```

## 📋 Checklist Before Submitting PR

- [ ] Code follows style guide (Black, Flake8 for Python)
- [ ] All new functions have docstrings
- [ ] Added unit tests for new functionality
- [ ] All tests pass: `pytest tests/`
- [ ] No hardcoded secrets or API keys
- [ ] Updated documentation if needed
- [ ] Commit messages are clear and descriptive
- [ ] Branch is up to date with main

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest Clarion-Backend/tests/ -v

# Run specific test file
pytest Clarion-Backend/tests/test_services.py -v

# Run with coverage
pytest Clarion-Backend/tests/ --cov=Clarion-Backend/

# Run only tests matching pattern
pytest -k "embedding" -v
```

### Writing Tests

```python
# tests/test_document_service.py
import pytest
from services.document_service import DocumentService

@pytest.fixture
def service():
    """Fixture to provide service instance."""
    return DocumentService()

def test_extract_text_from_pdf(service):
    """Test PDF text extraction."""
    result = service.extract_text('path/to/test.pdf')
    assert result is not None
    assert len(result) > 0

def test_invalid_file_raises_error(service):
    """Test error handling for invalid files."""
    with pytest.raises(InvalidFileError):
        service.extract_text('path/to/invalid.txt')
```

## 🐛 Bug Reports

### Before Submitting

- Check if bug already reported
- Verify it's reproducible
- Test with latest code
- Gather system information

### Bug Report Template

```markdown
## Description
Clear description of the bug

## Steps to Reproduce
1. Do this
2. Then this
3. Bug occurs

## Expected Behavior
What should happen

## Actual Behavior
What actually happened

## Screenshots/Logs
If applicable, add error logs or screenshots

## Environment
- OS: [e.g., Windows 10, Ubuntu 22.04]
- Python: [e.g., 3.10.5]
- Browser: [e.g., Chrome 120]
```

## 💡 Feature Requests

### Template

```markdown
## Description
Clear description of the feature

## Motivation
Why is this feature needed?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches you considered

## Additional Context
Any other relevant information
```

## 📚 Documentation Contributing

### Updating Docs

1. Clone repo and create feature branch
2. Edit markdown files in `/docs` or `/README.md`
3. Check that changes render correctly
4. Submit PR with documentation changes

### Documentation Style

- Use clear, concise language
- Add code examples where relevant
- Include links to related docs
- Keep technical level appropriate

## 🔍 Code Review Process

Your PR will be reviewed for:

- **Correctness**: Does it work as intended?
- **Style**: Does it follow project conventions?
- **Tests**: Are tests adequate?
- **Documentation**: Is it well-documented?
- **Security**: Any security concerns?

## 📈 Contribution Levels

### Level 1: Documentation & Typos
- Update README
- Fix typos
- Improve documentation

### Level 2: Small Fixes
- Bug fixes
- Code cleanup
- Small improvements

### Level 3: Features
- New functionality
- Significant refactoring
- Performance improvements

## 🏆 Recognition

Contributors will be:
- Added to CONTRIBUTORS.md
- Thanked in release notes
- Recognized in documentation

## 📞 Getting Help

- 💬 GitHub Discussions
- 🐛 GitHub Issues
- 📧 Email: [maintainer-email]
- 📖 Read the docs

## 🚀 Your First Contribution

Looking to make your first contribution? Look for issues labeled:
- `good first issue`
- `help wanted`
- `documentation`

## 📋 Development Tools Setup

```bash
# Install development dependencies
pip install pytest black flake8 mypy isort

# Format code
black Clarion-Backend/ frontend/

# Sort imports
isort Clarion-Backend/

# Lint code
flake8 Clarion-Backend/

# Type check
mypy Clarion-Backend/ --ignore-missing-imports

# Run tests
pytest Clarion-Backend/tests/ -v --cov
```

## 🎯 Priority Areas for Contribution

High priority areas where help is needed:

1. **Testing**: Improve test coverage
2. **Documentation**: Enhance docs and examples
3. **Performance**: Optimization suggestions
4. **Accessibility**: Frontend improvements
5. **Internationalization**: Multi-language support
6. **Features**: New analysis capabilities

## 📝 Changelog

Changelog entries should be added to `CHANGELOG.md` for each PR:

```markdown
## [Unreleased]

### Added
- New feature description

### Fixed
- Bug fix description

### Changed
- Change description for existing features
```

---

Thank you for contributing to Clarion! Your effort helps make this project better for everyone.
