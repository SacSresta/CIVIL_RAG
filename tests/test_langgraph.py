#!/usr/bin/env python3
"""Test script for LangGraph conversion"""

def test_imports():
    try:
        print("Testing basic imports...")
        from langchain_core.messages import HumanMessage, AIMessage
        print("✓ LangChain core imports successful")
        
        from langgraph.graph import StateGraph, END
        print("✓ LangGraph imports successful")
        
        from civic_rag.backend.chatmodels import get_agent
        print("✓ Chatmodels import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agent_creation():
    try:
        print("\nTesting agent creation...")
        from civic_rag.backend.chatmodels import get_agent
        agent = get_agent()
        print("✓ Agent creation successful")
        return agent
    except Exception as e:
        print(f"✗ Agent creation error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simple_query(agent):
    try:
        print("\nTesting simple query...")
        from langchain_core.messages import HumanMessage
        
        initial_state = {
            "messages": [HumanMessage(content="What are my rights during a protest?")],
            "context": "",
            "web_results": "",
            "rag_results": "",
            "final_answer": ""
        }
        
        result = agent.invoke(initial_state)
        print("✓ Query execution successful")
        print(f"Result type: {type(result)}")
        print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
        return result
    except Exception as e:
        print(f"✗ Query execution error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("LangGraph Conversion Test")
    print("=" * 50)
    
    if test_imports():
        agent = test_agent_creation()
        if agent:
            result = test_simple_query(agent)
            if result:
                print(f"\n✓ All tests passed!")
                print(f"Final answer: {result.get('final_answer', 'No final answer found')}")
    
    print("\nTest completed.")
