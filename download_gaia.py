#!/usr/bin/env python3
"""
GAIA Dataset Downloader
Enhanced version with better authentication and error handling
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any

def setup_huggingface_auth():
    """Setup HuggingFace authentication with multiple methods"""
    from huggingface_hub import login, HfApi
    
    # Method 1: Check if already authenticated
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Already authenticated as: {user_info['name']}")
        return True
    except Exception:
        pass
    
    # Method 2: Check for HF_TOKEN environment variable
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        try:
            login(token=hf_token)
            print("✅ Authenticated using HF_TOKEN environment variable")
            return True
        except Exception as e:
            print(f"❌ Failed to authenticate with HF_TOKEN: {e}")
    
    # Method 3: Interactive login
    print("🔐 Please authenticate with HuggingFace...")
    print("You can either:")
    print("1. Run: huggingface-cli login")
    print("2. Set HF_TOKEN environment variable")
    print("3. Login interactively now")
    
    response = input("Would you like to login interactively? (y/n): ").lower().strip()
    if response == 'y':
        try:
            login()
            print("✅ Interactive authentication successful")
            return True
        except Exception as e:
            print(f"❌ Interactive authentication failed: {e}")
    
    return False

def verify_gaia_format(sample_data: Dict[Any, Any]) -> bool:
    """Verify that the downloaded data matches expected GAIA format"""
    required_fields = [
        'Question',
        'Final answer', 
        'Level',
        'Annotator Metadata'
    ]
    
    optional_fields = [
        'file_name',
        'file_path',
        'task_id'
    ]
    
    # Check required fields
    for field in required_fields:
        if field not in sample_data:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Verify level format
    level = sample_data.get('Level', '')
    if not level.startswith('Level '):
        print(f"❌ Invalid level format: {level}")
        return False
    
    print("✅ Data format verification passed")
    return True

def download_gaia_dataset():
    """Main function to download GAIA dataset"""
    
    try:
        from datasets import load_dataset
        print("🔄 Loading datasets library...")
        
    except ImportError:
        print("❌ datasets library not found. Installing...")
        os.system("pip install datasets transformers huggingface_hub")
        from datasets import load_dataset
    
    # Setup authentication
    if not setup_huggingface_auth():
        print("❌ Authentication failed. Please ensure you have access to GAIA dataset.")
        print("Visit: https://huggingface.co/datasets/gaia-benchmark/GAIA")
        return False
    
    try:
        print("🔄 Downloading GAIA dataset...")
        
        # Load the dataset
        dataset = load_dataset("gaia-benchmark/GAIA")
        
        print(f"📊 Dataset loaded with splits: {list(dataset.keys())}")
        for split, data in dataset.items():
            print(f"   {split}: {len(data)} samples")
        
        # Verify format with first sample
        if len(dataset['validation']) > 0:
            sample = dataset['validation'][0]
            if not verify_gaia_format(sample):
                print("⚠️  Warning: Dataset format may not match expected GAIA format")
                print("Sample keys:", list(sample.keys()))
        
        # Setup save directory
        save_path = Path("/Users/linwen/Desktop/agent_AC/mybank_webthinker/dataset/GAIA/data")
        save_path.mkdir(exist_ok=True, parents=True)
        
        # Save each split as JSONL
        for split_name, split_data in dataset.items():
            output_file = save_path / f"{split_name}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"✅ Saved {split_name} split: {output_file} ({len(split_data)} samples)")
        
        # Save dataset info
        info_file = save_path / "dataset_info.json"
        dataset_info = {
            "source": "gaia-benchmark/GAIA",
            "splits": {split: len(data) for split, data in dataset.items()},
            "total_samples": sum(len(data) for data in dataset.values()),
            "description": "GAIA (General AI Assistants) benchmark dataset"
        }
        
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Dataset info saved: {info_file}")
        print("🎉 GAIA dataset download completed successfully!")
        
        # Print summary
        print("\n📈 Summary:")
        for split, data in dataset.items():
            print(f"   {split.capitalize()}: {len(data)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Ensure you have requested access: https://huggingface.co/datasets/gaia-benchmark/GAIA")
        print("2. Verify your HuggingFace authentication:")
        print("   - Run: huggingface-cli login")
        print("   - Or set: export HF_TOKEN='your_token_here'")
        print("3. Check your internet connection")
        print("4. Verify you accepted the dataset terms of use")
        return False

if __name__ == "__main__":
    success = download_gaia_dataset()
    sys.exit(0 if success else 1)
