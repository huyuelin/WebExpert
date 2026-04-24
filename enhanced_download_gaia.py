#!/usr/bin/env python3
"""
Enhanced GAIA Dataset Downloader with Better Authentication and Error Handling
"""

import os
import json
from pathlib import Path
from datasets import load_dataset
import huggingface_hub
from huggingface_hub import login, whoami
import getpass

def check_authentication():
    """Check if user is authenticated with HuggingFace"""
    try:
        user_info = whoami()
        print(f"✅ Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not authenticated with HuggingFace")
        return False

def authenticate():
    """Authenticate with HuggingFace using multiple methods"""
    print("🔑 Setting up HuggingFace authentication...")
    
    # Method 1: Check environment variable
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            print("✅ Authenticated using HF_TOKEN environment variable")
            return True
        except Exception as e:
            print(f"❌ Failed to authenticate with environment token: {e}")
    
    # Method 2: Interactive login
    print("\n🔗 Please visit: https://huggingface.co/settings/tokens")
    print("📝 Create a new token with 'Read' access")
    print("🔐 Then enter your token below:")
    
    token = getpass.getpass("Enter your HuggingFace token: ")
    
    if token.strip():
        try:
            login(token=token.strip())
            print("✅ Authentication successful!")
            return True
        except Exception as e:
            print(f"❌ Authentication failed: {e}")
            return False
    else:
        print("❌ No token provided")
        return False

def check_gaia_access():
    """Test access to GAIA dataset"""
    try:
        print("🔍 Testing GAIA dataset access...")
        # Try to load just metadata to test access
        dataset_info = huggingface_hub.dataset_info("gaia-benchmark/GAIA")
        print("✅ GAIA dataset access confirmed")
        print(f"📊 Dataset info: {dataset_info.description[:100]}...")
        return True
    except Exception as e:
        print(f"❌ Cannot access GAIA dataset: {e}")
        print("\n🚨 Access Issues:")
        print("1. Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA")
        print("2. Click 'Request access' and wait for approval")
        print("3. Make sure you're authenticated with the approved account")
        return False

def download_gaia_dataset():
    """Download the real GAIA dataset"""
    try:
        print("📥 Downloading GAIA dataset...")
        
        # Load the dataset
        dataset = load_dataset("gaia-benchmark/GAIA", name="2023_all")
        
        print("✅ Dataset downloaded successfully!")
        print(f"📊 Dataset splits: {list(dataset.keys())}")
        
        # Show sample statistics
        for split_name, split_data in dataset.items():
            print(f"🔍 {split_name}: {len(split_data)} examples")
            
        # Create output directory
        output_dir = Path("dataset/GAIA/official")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each split as JSONL
        for split_name, split_data in dataset.items():
            output_file = output_dir / f"gaia_{split_name}.jsonl"
            
            print(f"💾 Saving {split_name} to {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
                    
            print(f"✅ Saved {len(split_data)} items to {output_file}")
            
        # Show sample question
        if 'test' in dataset:
            sample = dataset['test'][0]
            print("\n🎯 Sample GAIA Question:")
            print(f"Question: {sample.get('Question', 'N/A')[:200]}...")
            print(f"Level: {sample.get('Level', 'N/A')}")
            print(f"Final Answer: {sample.get('Final answer', 'N/A')[:100]}...")
            
        print("\n🎉 GAIA dataset successfully downloaded!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download GAIA dataset: {e}")
        return False

def main():
    """Main function to orchestrate GAIA dataset download"""
    print("🌟 Enhanced GAIA Dataset Downloader")
    print("=" * 50)
    
    # Step 1: Check authentication
    if not check_authentication():
        print("\n🔑 Authentication required...")
        if not authenticate():
            print("❌ Cannot proceed without authentication")
            return False
    
    # Step 2: Check dataset access
    if not check_gaia_access():
        print("❌ Cannot proceed without dataset access")
        return False
    
    # Step 3: Download dataset
    success = download_gaia_dataset()
    
    if success:
        print("\n" + "="*50)
        print("🎉 SUCCESS! Real GAIA dataset is now available")
        print("📁 Location: dataset/GAIA/official/")
        print("🔄 You can now re-run the expert experience extraction pipeline")
        print("🚀 Run: python run_extraction.py --dataset GAIA")
        print("="*50)
    else:
        print("\n❌ Download failed. Please check the errors above and try again.")
    
    return success

if __name__ == "__main__":
    main()