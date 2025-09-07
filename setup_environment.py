import subprocess
import sys
import os

def install_package(package):
    """Install a package with error handling"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Setup the environment with proper package versions"""
    print("🔧 Setting up Enhanced Crypto Trading System Environment")
    print("=" * 60)
    
    # Core packages
    packages = [
        "pandas>=1.5.0,<2.0.0",
        "numpy>=1.21.0,<2.0.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.1.0,<1.4.0",
        "joblib>=1.2.0",
        "tqdm>=4.64.0",
        "python-dateutil>=2.8.0",
        "pytz>=2022.1"
    ]
    
    print("\n📦 Installing core packages...")
    for package in packages:
        install_package(package)
    
    # Try pandas-ta with compatibility fix
    print("\n🔧 Attempting to install pandas-ta...")
    
    # First try to install with specific numpy version
    if install_package("numpy==1.24.3"):
        if install_package("pandas-ta==0.3.14b0"):
            print("✅ pandas-ta installed successfully")
        else:
            print("❌ pandas-ta installation failed, but manual indicators will be used")
    
    # Optional packages
    optional_packages = [
        "imbalanced-learn>=0.9.0",
        "optuna>=3.0.0"
    ]
    
    print("\n📦 Installing optional packages...")
    for package in optional_packages:
        install_package(package)
    
    print("\n✅ Environment setup completed!")
    print("You can now run: python crypto_trading_app.py")

if __name__ == "__main__":
    main()