#!/bin/bash

# Bias Detection Models - Simple Setup Script

echo "🚀 Setting up Bias Detection Models..."
echo "====================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip first."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Install requirements
echo "📚 Installing dependencies from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
    echo "✅ Dependencies installed successfully"
else
    echo "❌ requirements.txt not found in current directory"
    exit 1
fi

# Download NLTK data
echo "📥 Downloading NLTK data..."
if [ -f "download.py" ]; then
    python3 download.py
    echo "✅ NLTK data downloaded successfully"
else
    echo "⚠️  download.py not found, downloading NLTK data manually..."
    python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
fi

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 Usage:"
echo "1. Navigate to a domain: cd comp/ (or econ/, tech/, business/)"
echo "2. Generate test papers: python split_dataset.py"
echo "3. Run predictions: python evaluate.py"
echo ""
echo "📖 See README.md for detailed instructions"
