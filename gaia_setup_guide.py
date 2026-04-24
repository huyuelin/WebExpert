#!/usr/bin/env python3
"""
HuggingFace GAIA Dataset Setup Guide and Diagnostic Tool
"""

import os
import sys
from pathlib import Path
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    required_packages = ['datasets', 'huggingface_hub', 'torch', 'transformers']
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n🚨 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    return True

def check_authentication():
    """Check HuggingFace authentication status"""
    print("\n🔑 Checking HuggingFace authentication...")
    
    try:
        from huggingface_hub import whoami
        user_info = whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception as e:
        print(f"❌ Not authenticated: {e}")
        return False

def show_authentication_guide():
    """Show step-by-step authentication guide"""
    print("\n" + "="*60)
    print("🔐 HUGGINGFACE AUTHENTICATION SETUP GUIDE")
    print("="*60)
    print()
    print("📋 Step 1: Get HuggingFace Token")
    print("   • Visit: https://huggingface.co/settings/tokens")
    print("   • Click 'New token'")
    print("   • Name: 'GAIA Dataset Access'")
    print("   • Type: Read")
    print("   • Click 'Generate a token'")
    print("   • Copy the token (starts with 'hf_...')")
    print()
    print("📋 Step 2: Set Environment Variable")
    print("   • For current session:")
    print("     export HF_TOKEN='your_token_here'")
    print("   • For permanent setup (add to ~/.bashrc or ~/.zshrc):")
    print("     echo 'export HF_TOKEN=\"your_token_here\"' >> ~/.bashrc")
    print()
    print("📋 Step 3: Alternative - Use CLI login")
    print("   huggingface-cli login")
    print("   # Enter your token when prompted")
    print()
    print("📋 Step 4: Request GAIA Dataset Access")
    print("   • Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA")
    print("   • Click 'Request access' button")
    print("   • Wait for approval (may take some time)")
    print()
    print("=" * 60)

def test_gaia_access():
    """Test if GAIA dataset can be accessed"""
    print("\n🧪 Testing GAIA dataset access...")
    
    try:
        from datasets import load_dataset_builder
        builder = load_dataset_builder("gaia-benchmark/GAIA", name="2023_all")
        print("✅ GAIA dataset is accessible")
        return True
    except Exception as e:
        print(f"❌ Cannot access GAIA dataset: {e}")
        if "gated dataset" in str(e).lower():
            print("🚨 Dataset is gated - you need to request access")
            print("Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA")
        elif "authentication" in str(e).lower():
            print("🚨 Authentication required")
        return False

def check_current_data():
    """Check what GAIA data currently exists"""
    print("\n📁 Checking current GAIA data...")
    
    gaia_dir = Path("dataset/GAIA")
    if not gaia_dir.exists():
        print("❌ No GAIA dataset directory found")
        return
    
    print(f"📂 GAIA directory: {gaia_dir}")
    
    # Check for different data sources
    sample_file = gaia_dir / "sample_data.json"
    official_dir = gaia_dir / "official"
    
    if sample_file.exists():
        print(f"📄 Found sample data: {sample_file}")
        try:
            import json
            with open(sample_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"   🔢 Contains {len(data)} items")
            if data:
                first_item = data[0]
                question = first_item.get('Question', first_item.get('question', 'N/A'))
                print(f"   🎯 Sample: {question[:100]}...")
                
                # Detect language
                if any('\u4e00' <= char <= '\u9fff' for char in question):
                    print("   🌍 Language: Chinese (Sample data)")
                else:
                    print("   🌍 Language: English")
        except Exception as e:
            print(f"   ❌ Error reading sample data: {e}")
    
    if official_dir.exists():
        print(f"📂 Found official directory: {official_dir}")
        jsonl_files = list(official_dir.glob("*.jsonl"))
        if jsonl_files:
            print(f"   📄 JSONL files: {[f.name for f in jsonl_files]}")
        else:
            print("   📄 No JSONL files found")
    else:
        print("❌ No official GAIA data directory found")

def show_next_steps():
    """Show what to do next"""
    print("\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    print()
    print("1️⃣ Set up authentication (if not done):")
    print("   export HF_TOKEN='your_token_here'")
    print("   # OR")
    print("   huggingface-cli login")
    print()
    print("2️⃣ Request GAIA dataset access:")
    print("   Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA")
    print("   Click 'Request access' and wait for approval")
    print()
    print("3️⃣ Download the real dataset:")
    print("   python enhanced_download_gaia.py")
    print("   # OR manually with datasets library")
    print()
    print("4️⃣ Update the pipeline to use real data:")
    print("   python run_extraction.py --dataset GAIA")
    print()
    print("=" * 60)

def main():
    """Main diagnostic function"""
    print("🛠️  GAIA Dataset Setup Diagnostic Tool")
    print("="*50)
    
    # Check 1: Dependencies
    deps_ok = check_dependencies()
    
    # Check 2: Authentication
    auth_ok = check_authentication()
    
    # Check 3: Current data
    check_current_data()
    
    # Check 4: GAIA access (only if authenticated)
    access_ok = False
    if auth_ok:
        access_ok = test_gaia_access()
    
    # Summary and guidance
    print("\n" + "="*50)
    print("📊 DIAGNOSTIC SUMMARY")
    print("="*50)
    print(f"Dependencies: {'✅ OK' if deps_ok else '❌ MISSING'}")
    print(f"Authentication: {'✅ OK' if auth_ok else '❌ NEEDED'}")
    print(f"GAIA Access: {'✅ OK' if access_ok else '❌ NEEDED'}")
    
    if not auth_ok:
        show_authentication_guide()
    
    show_next_steps()

if __name__ == "__main__":
    main()