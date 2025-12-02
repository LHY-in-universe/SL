# Contributing to splitlearn-comm

Thank you for your interest in contributing to splitlearn-comm! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

### Prerequisites

- Python >= 3.8
- Git
- Basic knowledge of gRPC and Protocol Buffers
- Familiarity with PyTorch

### Communication

- **GitHub Issues**: Bug reports, feature requests, questions
- **Pull Requests**: Code contributions
- **Discussions**: General discussions and ideas

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR-USERNAME/splitlearn-comm.git
cd splitlearn-comm

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL-OWNER/splitlearn-comm.git
```

### 2. Create Development Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"
```

### 3. Install Development Dependencies

The `[dev]` extra includes:
- pytest (testing)
- pytest-cov (coverage)
- black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

### 4. Compile Protocol Buffers

```bash
bash scripts/compile_proto.sh
```

Verify compilation:
```bash
ls src/splitlearn_comm/protocol/generated/
# Should see: compute_service_pb2.py, compute_service_pb2_grpc.py, etc.
```

---

## Project Structure

```
splitlearn-comm/
├── src/splitlearn_comm/          # Main package
│   ├── core/                      # Core abstractions
│   │   ├── compute_function.py    # ComputeFunction interface
│   │   └── tensor_codec.py        # Tensor serialization
│   ├── protocol/                  # gRPC protocol
│   │   ├── protos/                # .proto files
│   │   └── generated/             # Generated code
│   ├── server/                    # Server implementation
│   │   ├── servicer.py            # gRPC servicer
│   │   └── grpc_server.py         # Server wrapper
│   ├── client/                    # Client implementation
│   │   ├── retry.py               # Retry strategies
│   │   └── grpc_client.py         # Client wrapper
│   └── __init__.py                # Package exports
├── tests/                         # Test files
├── examples/                      # Example scripts
├── docs/                          # Documentation
├── scripts/                       # Build/utility scripts
├── pyproject.toml                 # Package configuration
└── README.md                      # Main documentation
```

---

## Making Changes

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Your Changes

Follow these guidelines:

**For New Features:**
1. Add code in appropriate module
2. Add tests in `tests/`
3. Update documentation in `docs/`
4. Add example if applicable
5. Update `README.md` if needed

**For Bug Fixes:**
1. Add test that reproduces the bug
2. Fix the bug
3. Verify test passes
4. Update documentation if needed

**For Documentation:**
1. Update relevant `.md` files
2. Ensure examples are correct
3. Check for broken links

### 3. Code Style

We follow PEP 8 with some modifications:

```bash
# Format code with black
black src/ tests/

# Sort imports with isort
isort src/ tests/

# Check with flake8
flake8 src/ tests/

# Type check with mypy
mypy src/
```

**Style Guidelines:**
- Line length: 88 characters (black default)
- Use type hints for all functions
- Docstrings for all public APIs
- Clear variable names
- Comments for complex logic

**Example:**

```python
def compute(
    self,
    input_tensor: torch.Tensor,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Perform computation on input tensor.

    Args:
        input_tensor: Input tensor to process
        device: Target device for computation

    Returns:
        Computed output tensor

    Raises:
        ValueError: If input shape is invalid
    """
    if input_tensor.dim() < 2:
        raise ValueError(f"Expected 2D+ tensor, got {input_tensor.dim()}D")

    # Move to device and compute
    x = input_tensor.to(device)
    return self.model(x)
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=splitlearn_comm --cov-report=html

# Run specific test file
pytest tests/test_client.py

# Run specific test
pytest tests/test_client.py::test_connection
```

### Writing Tests

Place tests in `tests/` directory:

```python
# tests/test_custom_feature.py
import pytest
import torch
from splitlearn_comm import GRPCComputeClient, GRPCComputeServer
from splitlearn_comm.core import ComputeFunction

class DummyFunction(ComputeFunction):
    def compute(self, input_tensor):
        return input_tensor * 2

@pytest.fixture
def server():
    """Fixture for test server."""
    server = GRPCComputeServer(
        compute_fn=DummyFunction(),
        port=50099  # Use different port for tests
    )
    server.start()
    yield server
    server.stop()

def test_my_feature(server):
    """Test my new feature."""
    client = GRPCComputeClient("localhost:50099")
    assert client.connect()

    input_tensor = torch.randn(1, 10)
    output = client.compute(input_tensor)

    assert torch.allclose(output, input_tensor * 2)
    client.close()
```

### Test Coverage

Aim for >90% test coverage:
- All public APIs must have tests
- Edge cases should be covered
- Error conditions should be tested

---

## Submitting Changes

### 1. Commit Your Changes

```bash
# Stage changes
git add .

# Commit with clear message
git commit -m "Add feature: custom retry strategy

- Implement AdaptiveRetry class
- Add tests for retry logic
- Update documentation
"
```

**Commit Message Guidelines:**
- First line: Brief summary (<50 chars)
- Blank line
- Detailed description
- Reference issues: "Fixes #123"

### 2. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 3. Create Pull Request

1. Go to GitHub repository
2. Click "New Pull Request"
3. Select your branch
4. Fill in PR template:
   - **Description**: What does this PR do?
   - **Motivation**: Why is this change needed?
   - **Testing**: How was this tested?
   - **Checklist**: Complete all items

**PR Template:**

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- List of specific changes
- Another change

## Testing
How was this tested? Include test results.

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] No breaking changes (or documented)
```

### 4. Code Review

- Address reviewer feedback
- Push additional commits to same branch
- Be responsive and collaborative
- Don't take criticism personally

---

## Reporting Issues

### Bug Reports

Use this template:

```markdown
**Description**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Create server with...
2. Connect client to...
3. Call compute() with...
4. See error

**Expected Behavior**
What you expected to happen

**Actual Behavior**
What actually happened

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python version: [e.g., 3.9.7]
- splitlearn-comm version: [e.g., 1.0.0]
- PyTorch version: [e.g., 2.0.1]

**Additional Context**
Error messages, stack traces, logs
```

### Feature Requests

Use this template:

```markdown
**Feature Description**
What feature would you like to see?

**Use Case**
What problem would this solve?

**Proposed Solution**
How might this be implemented?

**Alternatives Considered**
What alternatives have you considered?
```

---

## Development Tips

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use built-in debugger
import pdb; pdb.set_trace()
```

### Testing Locally

```bash
# Terminal 1: Start server
python examples/simple_server.py

# Terminal 2: Run client
python examples/simple_client.py
```

### Regenerating Protobuf

If you modify `.proto` files:

```bash
bash scripts/compile_proto.sh
```

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
client.compute(input_tensor)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

---

## Best Practices

1. **Write tests first** (TDD when possible)
2. **Keep PRs focused** (one feature/fix per PR)
3. **Update documentation** (docs are code too)
4. **Ask questions** (no question is too small)
5. **Be respectful** (follow code of conduct)

---

## Release Process

(For maintainers)

1. Update version in `src/splitlearn_comm/__version__.py`
2. Update CHANGELOG.md
3. Create git tag: `git tag -a v1.x.x -m "Release v1.x.x"`
4. Push tag: `git push origin v1.x.x`
5. Build package: `python -m build`
6. Upload to PyPI: `twine upload dist/*`

---

## Questions?

- Open an issue for questions
- Check existing issues and PRs
- Read the documentation

Thank you for contributing! 🎉
