# Bias Detection Models - Simple Setup Script for Windows PowerShell

Write-Host "Setting up Bias Detection Models..." -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
$pythonCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonCmd = "python"
} elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    $pythonCmd = "python3"
} else {
    Write-Host "ERROR: Python 3 is not installed. Please install Python 3.8+ first." -ForegroundColor Red
    exit 1
}

$pythonVersion = & $pythonCmd --version 2>&1
Write-Host "Python found: $pythonVersion" -ForegroundColor Green

# Check if pip is installed
$pipCmd = $null
if (Get-Command pip -ErrorAction SilentlyContinue) {
    $pipCmd = "pip"
} elseif (Get-Command pip3 -ErrorAction SilentlyContinue) {
    $pipCmd = "pip3"
} else {
    Write-Host "ERROR: pip is not installed. Please install pip first." -ForegroundColor Red
    exit 1
}

$pipVersion = & $pipCmd --version 2>&1
Write-Host "pip found: $pipVersion" -ForegroundColor Green
Write-Host ""

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "WARNING: Virtual environment already exists. Using existing venv." -ForegroundColor Yellow
} else {
    & $pythonCmd -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully" -ForegroundColor Green
}
Write-Host ""

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "ERROR: Failed to activate virtual environment" -ForegroundColor Red
    Write-Host "Activation script not found at: $activateScript" -ForegroundColor Red
    Write-Host "Trying to continue anyway..." -ForegroundColor Yellow
}
Write-Host ""

# Upgrade pip in venv
Write-Host "Upgrading pip..." -ForegroundColor Yellow
$venvPython = Join-Path $PSScriptRoot "venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    & $venvPython -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: pip upgrade may have failed, but continuing..." -ForegroundColor Yellow
    }
} else {
    pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: pip upgrade may have failed, but continuing..." -ForegroundColor Yellow
    }
}
Write-Host ""

# Install requirements
Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
if (Test-Path "requirements.txt") {
    if (Test-Path $venvPython) {
        & $venvPython -m pip install -r requirements.txt
    } else {
        pip install -r requirements.txt
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Dependencies installed successfully" -ForegroundColor Green
    } else {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "ERROR: requirements.txt not found in current directory" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Download NLTK data
Write-Host "Downloading NLTK data..." -ForegroundColor Yellow
$pythonCmdForNLTK = if (Test-Path $venvPython) { $venvPython } else { "python" }
if (Test-Path "download.py") {
    & $pythonCmdForNLTK download.py
    if ($LASTEXITCODE -eq 0) {
        Write-Host "NLTK data downloaded successfully" -ForegroundColor Green
    } else {
        Write-Host "WARNING: NLTK download may have failed, trying manual download..." -ForegroundColor Yellow
        & $pythonCmdForNLTK -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
    }
} else {
    Write-Host "WARNING: download.py not found, downloading NLTK data manually..." -ForegroundColor Yellow
    & $pythonCmdForNLTK -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')"
}
Write-Host ""

Write-Host ""
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==================" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:" -ForegroundColor Cyan
Write-Host "1. Activate virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   (If you get an execution policy error, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser)" -ForegroundColor Yellow
Write-Host "2. Navigate to a domain: cd comp\ (or econ\, tech\, business\, health_science\)" -ForegroundColor White
Write-Host "3. Generate test papers: python split_dataset.py" -ForegroundColor White
Write-Host "4. Run predictions: python evaluate.py" -ForegroundColor White
Write-Host ""
Write-Host "Note: Remember to activate the virtual environment before running any scripts" -ForegroundColor Yellow
Write-Host "   PowerShell: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "   Command Prompt: venv\Scripts\activate.bat" -ForegroundColor White
Write-Host ""
Write-Host "See README.md for detailed instructions" -ForegroundColor Cyan
