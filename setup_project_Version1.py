"""
Project setup script for Enhanced Crypto Trading System
User: amenson1983new
Date: 2025-08-31 16:18:54 UTC
"""

import os
import json
from pathlib import Path

def create_directory_structure():
    """Create the complete directory structure"""
    
    directories = [
        "config",
        "data/raw",
        "data/processed", 
        "data/predictions",
        "models",
        "src/core",
        "src/utils",
        "src/trading",
        "examples",
        "notebooks",
        "tests",
        "logs",
        "reports",
        "scripts"
    ]
    
    print("🏗️  Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep files for empty directories
        gitkeep_path = Path(directory) / ".gitkeep"
        if not any(Path(directory).iterdir()):
            gitkeep_path.touch()
    
    print("✅ Directory structure created!")

def create_init_files():
    """Create __init__.py files for Python packages"""
    
    init_files = [
        "src/__init__.py",
        "src/core/__init__.py", 
        "src/utils/__init__.py",
        "src/trading/__init__.py",
        "examples/__init__.py",
        "tests/__init__.py"
    ]
    
    print("📝 Creating __init__.py files...")
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ __init__.py files created!")

def create_gitignore():
    """Create comprehensive .gitignore file"""
    
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# Data files
*.csv
*.h5
*.pkl
*.json
!config/*.json
!**/requirements.txt

# Model files
models/*.h5
models/*.pkl

# Log files
logs/*.log

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Custom
temp/
output/
checkpoints/
wandb/
mlruns/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content.strip())
    
    print("✅ .gitignore created!")

def create_readme():
    """Create comprehensive README.md"""
    
    readme_content = """# Enhanced Crypto Trading Predictor System

**Created by:** amenson1983new  
**Date:** 2025-08-31 16:18:54 UTC  
**Target:** 85% profitable trades on BNBUSDT

## 🎯 Overview

Advanced cryptocurrency trading prediction system that combines Deep Learning (LSTM/GRU) with Machine Learning ensemble methods to predict BUY/SELL/SKIP signals with dynamic stop-loss and take-profit calculations.

## 🚀 Features

- **Hybrid Architecture:** LSTM + GRU + CNN + Ensemble ML
- **75+ Technical Indicators:** RSI, MACD, Bollinger Bands, Ichimoku, etc.
- **Advanced Risk Management:** Dynamic position sizing, stop-loss calculation
- **Multi-timeframe Analysis:** 5, 10, 20, 50 period features
- **Hyperparameter Optimization:** Optuna integration
- **Real-time Predictions:** Live trading signal generation
- **Comprehensive Backtesting:** Performance analysis framework

## 📦 Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd crypto_trading_system