#!/usr/bin/env python3
"""Minimal test"""

def test_config():
    try:
        import civic_rag.config as config
        print(f"GROQ_API_KEY: {'SET' if config.GROQ_API_KEY else 'NOT SET'}")
        print(f"First 10 chars: {config.GROQ_API_KEY[:10] if config.GROQ_API_KEY else 'None'}")
        return bool(config.GROQ_API_KEY)
    except Exception as e:
        print(f"Config error: {e}")
        return False

def test_basic_import():
    try:
        from langchain_groq import ChatGroq
        print("✓ ChatGroq import successful")
        return True
    except Exception as e:
        print(f"✗ ChatGroq import failed: {e}")
        return False

if __name__ == "__main__":
    print("Minimal Test")
    print("=" * 20)
    
    if test_config():
        test_basic_import()
    else:
        print("Skipping other tests due to config issues")
