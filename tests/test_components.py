#!/usr/bin/env python3
"""Simple test for LangGraph components"""

import sys
import os

def test_simple_llm():
    try:
        print("Testing Groq LLM...")
        from langchain_groq import ChatGroq
        import civic_rag.config as config
        
        llm = ChatGroq(
            model='meta-llama/llama-4-maverick-17b-128e-instruct',
            temperature=0.8,
            api_key=config.GROQ_API_KEY
        )
        
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Hello, can you hear me?")])
        print("✓ LLM test successful")
        print(f"Response: {response.content[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ LLM test failed: {e}")
        return False

def test_tools():
    try:
        print("\nTesting tools...")
        from civic_rag.backend.chatmodels import web_search, rag_search
        
        # Test web search
        print("Testing web search...")
        web_result = web_search.invoke("Nepal protest news")
        print(f"✓ Web search successful: {len(web_result)} characters")
        
        # Test RAG search (might fail if no data)
        print("Testing RAG search...")
        try:
            rag_result = rag_search.invoke("protest rights")
            print(f"✓ RAG search successful: {len(rag_result)} characters")
        except Exception as e:
            print(f"⚠ RAG search failed (expected if no data): {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Tools test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Simple LangGraph Component Test")
    print("=" * 50)
    
    if test_simple_llm():
        test_tools()
    
    print("\nTest completed.")
