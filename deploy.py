#!/usr/bin/env python3
"""
CrediTrust Financial RAG System - Deployment Script
Automates setup and deployment of the complaint analysis system
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_command(command, description, cwd=None):
    """Run shell command with error handling"""
    print(f"🔧 {description}...")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=cwd
        )
        print(f"   ✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error details: {e.stderr}")
        return False

def check_requirements():
    """Check if all requirements are satisfied"""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        print(f"   ❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
        return False
    print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, '-m', 'pip', '--version'], 
                      check=True, capture_output=True)
        print("   ✅ pip is available")
    except subprocess.CalledProcessError:
        print("   ❌ pip is not available")
        return False
    
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    
    # Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )
    
    if success:
        print("   ✅ All dependencies installed successfully")
    else:
        print("   ❌ Failed to install some dependencies")
        print("   💡 Try: pip install --upgrade pip")
        
    return success

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directory structure...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'vector_store',
        'reports',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("   ✅ Directory structure created")
    return True

def download_sample_data():
    """Download or prepare sample data"""
    print("📊 Preparing sample data...")
    
    data_file = Path('data/raw/complaints.csv')
    
    if data_file.exists():
        print(f"   ✅ Data file exists: {data_file}")
        return True
    else:
        print("   ⚠️  CFPB complaint dataset not found")
        print("   📝 Please download from: https://www.consumerfinance.gov/data-research/consumer-complaints/")
        print("   📁 Place as: data/raw/complaints.csv")
        return False

def run_data_pipeline():
    """Execute the complete data pipeline"""
    print("⚙️ Running data processing pipeline...")
    
    # Task 1: Data preprocessing
    success = run_command(
        f"{sys.executable} src/task1_eda_preprocessing.py",
        "Task 1: Data preprocessing"
    )
    
    if not success:
        return False
    
    # Task 2: Vector store creation
    success = run_command(
        f"{sys.executable} src/task2_embedding_pipeline.py", 
        "Task 2: Vector store creation"
    )
    
    if not success:
        return False
    
    # Task 3: RAG evaluation
    success = run_command(
        f"{sys.executable} src/task3_rag_pipeline.py",
        "Task 3: RAG pipeline evaluation"
    )
    
    return success

def test_system():
    """Run system tests"""
    print("🧪 Running system tests...")
    
    success = run_command(
        f"{sys.executable} test_system.py",
        "System validation tests"
    )
    
    return success

def deploy_interface(interface_type="gradio", port=7860):
    """Deploy the web interface"""
    print(f"🚀 Deploying {interface_type} interface on port {port}...")
    
    if interface_type == "gradio":
        command = f"{sys.executable} app.py --interface gradio --port {port}"
    elif interface_type == "streamlit":
        command = f"streamlit run app.py --server.port {port}"
    else:
        print(f"   ❌ Unknown interface type: {interface_type}")
        return False
    
    print(f"   🌐 Starting server: {command}")
    print(f"   🔗 Access at: http://localhost:{port}")
    print(f"   🛑 Press Ctrl+C to stop")
    
    try:
        subprocess.run(command, shell=True, check=True)
    except KeyboardInterrupt:
        print("\n   🛑 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Server failed to start: {e}")
        return False
    
    return True

def main():
    """Main deployment workflow"""
    parser = argparse.ArgumentParser(description="CrediTrust RAG System Deployment")
    parser.add_argument("--mode", choices=["setup", "pipeline", "test", "deploy", "full"], 
                       default="full", help="Deployment mode")
    parser.add_argument("--interface", choices=["gradio", "streamlit"], 
                       default="gradio", help="Web interface type")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--skip-data", action="store_true", 
                       help="Skip data pipeline (use existing)")
    
    args = parser.parse_args()
    
    print("🏦 CrediTrust Financial RAG System - Deployment")
    print("=" * 60)
    
    if args.mode in ["setup", "full"]:
        if not check_requirements():
            print("❌ System requirements not met. Exiting.")
            return False
        
        if not install_dependencies():
            print("❌ Failed to install dependencies. Exiting.")
            return False
        
        setup_directories()
        
        if not download_sample_data() and not args.skip_data:
            print("❌ Data preparation failed. Download dataset manually.")
            return False
    
    if args.mode in ["pipeline", "full"] and not args.skip_data:
        if not run_data_pipeline():
            print("❌ Data pipeline failed. Check logs.")
            return False
    
    if args.mode in ["test", "full"]:
        if not test_system():
            print("⚠️  System tests completed with warnings.")
    
    if args.mode in ["deploy", "full"]:
        print("\n" + "="*60)
        print("🎉 DEPLOYMENT READY!")
        print("="*60)
        
        deploy_interface(args.interface, args.port)
    
    return True

if __name__ == "__main__":
    main()