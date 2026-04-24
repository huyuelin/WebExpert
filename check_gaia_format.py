#!/usr/bin/env python3
"""
GAIA Dataset Format Checker
Compares existing sample data with expected GAIA format
"""

import json
from pathlib import Path

def load_sample_data():
    """Load the existing sample data"""
    sample_file = Path("/Users/linwen/Desktop/agent_AC/mybank_webthinker/dataset/GAIA/sample_data.json")
    
    if not sample_file.exists():
        print("❌ No sample data found")
        return None
    
    with open(sample_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📄 Loaded sample data with {len(data)} items")
    return data

def analyze_format(data):
    """Analyze the format of sample data"""
    print("\n🔍 ANALYZING SAMPLE DATA FORMAT:")
    print("=" * 40)
    
    if not data or len(data) == 0:
        print("❌ No data to analyze")
        return
    
    # Analyze first sample
    sample = data[0]
    print(f"Sample keys: {list(sample.keys())}")
    
    # Check against expected GAIA format
    expected_gaia_keys = [
        'Question',
        'Final answer', 
        'Level',
        'Annotator Metadata'
    ]
    
    optional_gaia_keys = [
        'file_name',
        'file_path', 
        'task_id',
        'reasoning_over_question'
    ]
    
    print(f"\n📊 FORMAT ANALYSIS:")
    print("-" * 30)
    
    # Check required keys
    missing_required = []
    present_required = []
    
    for key in expected_gaia_keys:
        if key in sample:
            present_required.append(key)
            print(f"✅ {key}: Present")
        else:
            missing_required.append(key)
            print(f"❌ {key}: Missing")
    
    # Check optional keys
    present_optional = []
    for key in optional_gaia_keys:
        if key in sample:
            present_optional.append(key)
            print(f"🔸 {key}: Present (optional)")
    
    # Check extra keys not in GAIA format
    all_expected = set(expected_gaia_keys + optional_gaia_keys)
    extra_keys = [key for key in sample.keys() if key not in all_expected]
    
    if extra_keys:
        print(f"\n⚠️  Extra keys not in GAIA format: {extra_keys}")
    
    print(f"\n📈 SUMMARY:")
    print(f"   Required fields present: {len(present_required)}/{len(expected_gaia_keys)}")
    print(f"   Optional fields present: {len(present_optional)}")
    print(f"   Extra fields: {len(extra_keys)}")
    
    # Determine if this looks like real GAIA data
    if len(missing_required) == 0:
        print("✅ FORMAT MATCHES: Appears to be real GAIA format")
    else:
        print("❌ FORMAT MISMATCH: This appears to be sample/mock data")
        print("   Real GAIA data should be downloaded from HuggingFace")
    
    return len(missing_required) == 0

def show_sample_content(data):
    """Show sample content for inspection"""
    print(f"\n📝 SAMPLE CONTENT PREVIEW:")
    print("=" * 40)
    
    if data and len(data) > 0:
        sample = data[0]
        
        # Show question
        question = sample.get('Question', 'N/A')
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")
        
        # Show level
        level = sample.get('Level', 'N/A')
        print(f"Level: {level}")
        
        # Show answer preview
        answer = sample.get('Final answer', 'N/A')
        print(f"Answer: {answer[:100]}..." if len(str(answer)) > 100 else f"Answer: {answer}")
        
        # Check if questions are in English (typical for real GAIA)
        if 'Question' in sample:
            question_text = sample['Question']
            # Simple heuristic: real GAIA questions are in English
            chinese_chars = sum(1 for c in question_text if '\u4e00' <= c <= '\u9fff')
            if chinese_chars > len(question_text) * 0.3:
                print("🇨🇳 Language: Appears to be Chinese (not typical GAIA format)")
            else:
                print("🇺🇸 Language: Appears to be English (typical GAIA format)")

def main():
    """Main function"""
    print("GAIA Dataset Format Checker")
    print("=" * 50)
    
    # Load sample data
    data = load_sample_data()
    if not data:
        return
    
    # Analyze format
    is_real_gaia = analyze_format(data)
    
    # Show sample content
    show_sample_content(data)
    
    # Provide recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    print("=" * 40)
    
    if is_real_gaia:
        print("✅ Your sample data appears to match GAIA format")
        print("   You may already have some GAIA data")
    else:
        print("❌ Your sample data doesn't match real GAIA format")
        print("   You need to download the real GAIA dataset:")
        print("   1. Setup HuggingFace authentication")
        print("   2. Request access to gaia-benchmark/GAIA")
        print("   3. Run: python download_gaia.py")

if __name__ == "__main__":
    main()