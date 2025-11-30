#!/bin/bash
# GraphWiz Ireland - One-Stop Setup Script
# Works with both UV and pip automatically

set -e

echo "=================================="
echo "  GraphWiz Ireland - Setup"
echo "=================================="
echo ""

# Check if UV is available
if command -v uv &> /dev/null; then
    USE_UV=true
    echo "✓ Using UV package manager (fast!)"
else
    USE_UV=false
    echo "✓ Using pip"
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version"

# Determine venv directory
if [ "$USE_UV" = true ]; then
    VENV_DIR=".venv"
else
    VENV_DIR="venv"
fi

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    if [ "$USE_UV" = true ]; then
        uv venv
    else
        python3 -m venv venv
    fi
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate venv
echo "→ Activating virtual environment..."
source $VENV_DIR/bin/activate

# Install dependencies
echo "→ Installing dependencies..."
if [ "$USE_UV" = true ]; then
    uv pip install -r requirements.txt -q
else
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
fi
echo "✓ Dependencies installed"

# Download spaCy model
echo "→ Downloading spaCy model..."
if [ "$USE_UV" = true ]; then
    uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl -q
else
    python -m spacy download en_core_web_sm --quiet 2>/dev/null || python -m spacy download en_core_web_sm
fi
echo "✓ spaCy model ready"

# Setup .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ .env file created"
fi

# Create directories
mkdir -p dataset/wikipedia_ireland
echo "✓ Data directories ready"

# Test imports
echo "→ Testing installation..."
python -c "import streamlit, groq, faiss, spacy, networkx; print('✓ All packages working')"

echo ""
echo "=================================="
echo "✅ Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Set GROQ_API_KEY in .env (already done)"
echo "2. Build knowledge base: python build_graphwiz.py"
echo "3. Launch app: streamlit run src/app.py"
echo ""
