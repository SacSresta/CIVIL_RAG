from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import civic_rag.config as config
from typing import TypedDict, Annotated, Sequence
from langchain_core.output_parsers import StrOutputParser
import operator

# Import utilities and tools from separate modules
from .utils import get_vector_store_info
from .tools_utils import rag_search, web_search

# Enhanced state definition with proper annotations
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    web_results: str
    rag_results: str
    economic_web_data: str
    political_web_data: str
    social_web_data: str
    economic_rag_data: str
    political_rag_data: str
    social_rag_data: str
    economic_analysis: str
    political_analysis: str
    social_analysis: str
    safety_analysis: str
    legal_analysis: str
    final_answer: str


# Parallel web search nodes for different aspects
def economic_web_search_node(state: AgentState) -> dict:
    """Searches for economic aspects of protests."""
    user_question = state["messages"][-1].content
    query = f"Nepal protest economic impact business disruption GDP {user_question}"
    return {"economic_web_data": web_search.invoke(query)}

def political_web_search_node(state: AgentState) -> dict:
    """Searches for political aspects of protests."""
    user_question = state["messages"][-1].content
    query = f"Nepal protest political parties government response policy {user_question}"
    return {"political_web_data": web_search.invoke(query)}

def social_web_search_node(state: AgentState) -> dict:
    """Searches for social and cultural aspects of protests."""
    user_question = state["messages"][-1].content
    query = f"Nepal protest social impact community safety cultural {user_question}"
    return {"social_web_data": web_search.invoke(query)}

# Parallel RAG search nodes for different aspects
def economic_rag_search_node(state: AgentState) -> dict:
    """Searches RAG for economic protest guidance."""
    user_question = state["messages"][-1].content
    query = f"economic impact business disruption financial {user_question}"
    return {"economic_rag_data": rag_search.invoke(query)}

def political_rag_search_node(state: AgentState) -> dict:
    """Searches RAG for political protest guidance."""
    user_question = state["messages"][-1].content
    query = f"political parties government policy legal rights {user_question}"
    return {"political_rag_data": rag_search.invoke(query)}

def social_rag_search_node(state: AgentState) -> dict:
    """Searches RAG for social and safety guidance."""
    user_question = state["messages"][-1].content
    query = f"social safety community cultural impact {user_question}"
    return {"social_rag_data": rag_search.invoke(query)}

# Merge node to combine web and RAG results
def merge_data_node(state: AgentState) -> dict:
    """Merges web and RAG search results."""
    # Combine all web results
    web_results = f"""
    Economic Web Data: {state.get("economic_web_data", "No data")}
    Political Web Data: {state.get("political_web_data", "No data")}
    Social Web Data: {state.get("social_web_data", "No data")}
    """
    
    # Combine all RAG results
    rag_results = f"""
    Economic RAG Data: {state.get("economic_rag_data", "No data")}
    Political RAG Data: {state.get("political_rag_data", "No data")}
    Social RAG Data: {state.get("social_rag_data", "No data")}
    """
    
    return {
        "web_results": web_results,
        "rag_results": rag_results
    }

# Analysis nodes for different aspects
def economic_analysis_node(state: AgentState) -> dict:
    """Analyzes economic aspects of the protest."""
    llm = ChatGroq(
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.7,
        api_key=config.GROQ_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are an economic analyst specializing in protest impacts. Analyze the economic aspects based on:
    
    User Question: {question}
    
    Current Economic Data: {economic_web_data}
    
    Historical Economic Guidance: {economic_rag_data}
    
    Provide analysis covering:
    1. Impact on businesses and commerce
    2. Effects on employment and daily wages
    3. Supply chain disruptions
    4. Tourism and service sector impacts
    5. Long-term economic consequences
    
    Be specific with numbers and percentages where available.
    """)
    
    user_question = state["messages"][-1].content
    chain = prompt | llm | StrOutputParser()
    
    economic_analysis = chain.invoke({
        "question": user_question,
        "economic_web_data": state.get("economic_web_data", "No current data"),
        "economic_rag_data": state.get("economic_rag_data", "No historical data")
    })
    
    return {"economic_analysis": economic_analysis}

def political_analysis_node(state: AgentState) -> dict:
    """Analyzes political aspects of the protest."""
    llm = ChatGroq(
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.7,
        api_key=config.GROQ_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a political analyst specializing in Nepal's political landscape. Analyze based on:
    
    User Question: {question}
    
    Current Political Data: {political_web_data}
    
    Historical Political Context: {political_rag_data}
    
    Provide analysis covering:
    1. Key political actors and their positions
    2. Government response and policies
    3. Opposition strategies
    4. Constitutional and legal frameworks
    5. Potential political outcomes
    
    Focus on factual analysis without bias.
    """)
    
    user_question = state["messages"][-1].content
    chain = prompt | llm | StrOutputParser()
    
    political_analysis = chain.invoke({
        "question": user_question,
        "political_web_data": state.get("political_web_data", "No current data"),
        "political_rag_data": state.get("political_rag_data", "No historical data")
    })
    
    return {"political_analysis": political_analysis}

def social_analysis_node(state: AgentState) -> dict:
    """Analyzes social and cultural aspects of the protest."""
    llm = ChatGroq(
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.7,
        api_key=config.GROQ_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a social analyst focusing on community impacts. Analyze based on:
    
    User Question: {question}
    
    Current Social Data: {social_web_data}
    
    Historical Social Context: {social_rag_data}
    
    Provide analysis covering:
    1. Community sentiment and participation
    2. Impact on different social groups
    3. Cultural and religious considerations
    4. Media coverage and public opinion
    5. Social cohesion and divisions
    
    Be sensitive to cultural nuances.
    """)
    
    user_question = state["messages"][-1].content
    chain = prompt | llm | StrOutputParser()
    
    social_analysis = chain.invoke({
        "question": user_question,
        "social_web_data": state.get("social_web_data", "No current data"),
        "social_rag_data": state.get("social_rag_data", "No historical data")
    })
    
    return {"social_analysis": social_analysis}

def safety_analysis_node(state: AgentState) -> dict:
    """Analyzes safety and security aspects."""
    llm = ChatGroq(
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.7,
        api_key=config.GROQ_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a safety and security expert. Based on all available data, provide safety analysis:
    
    Web Data: {web_results}
    RAG Data: {rag_results}
    
    Focus on:
    1. Current safety risks and hotspots
    2. Recommended safety precautions
    3. Emergency contacts and procedures
    4. Safe routes and areas
    5. Time-specific safety advice
    
    Prioritize citizen safety above all.
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    safety_analysis = chain.invoke({
        "web_results": state.get("web_results", ""),
        "rag_results": state.get("rag_results", "")
    })
    
    return {"safety_analysis": safety_analysis}

def legal_analysis_node(state: AgentState) -> dict:
    """Analyzes legal rights and implications."""
    llm = ChatGroq(
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.7,
        api_key=config.GROQ_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a legal expert on Nepal's protest laws. Based on available data, provide legal guidance:
    
    Web Data: {web_results}
    RAG Data: {rag_results}
    
    Cover:
    1. Constitutional rights to protest
    2. Legal limitations and restrictions
    3. Arrest procedures and rights
    4. Legal aid contacts
    5. Documentation recommendations
    
    Cite specific laws where applicable.
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    legal_analysis = chain.invoke({
        "web_results": state.get("web_results", ""),
        "rag_results": state.get("rag_results", "")
    })
    
    return {"legal_analysis": legal_analysis}

def final_synthesis_node(state: AgentState) -> dict:
    """Synthesizes all analyses into comprehensive guidance."""
    llm = ChatGroq(
        model='meta-llama/llama-4-maverick-17b-128e-instruct',
        temperature=0.7,
        api_key=config.GROQ_API_KEY
    )
    
    prompt = ChatPromptTemplate.from_template("""
    You are a senior protest guidance advisor. Synthesize all analyses into actionable guidance:
    
    User Question: {question}
    
    Economic Analysis: {economic_analysis}
    
    Political Analysis: {political_analysis}
    
    Social Analysis: {social_analysis}
    
    Safety Analysis: {safety_analysis}
    
    Legal Analysis: {legal_analysis}
    
    Provide a comprehensive response that:
    1. Directly addresses the user's question
    2. Integrates insights from all analyses
    3. Prioritizes safety and legal compliance
    4. Offers practical, actionable advice
    5. Includes relevant contacts and resources
    
    Structure your response with clear sections and bullet points for readability.
    """)
    
    user_question = state["messages"][-1].content
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "question": user_question,
        "economic_analysis": state.get("economic_analysis", ""),
        "political_analysis": state.get("political_analysis", ""),
        "social_analysis": state.get("social_analysis", ""),
        "safety_analysis": state.get("safety_analysis", ""),
        "legal_analysis": state.get("legal_analysis", "")
    })
    
    return {
        "final_answer": response,
        "messages": [AIMessage(content=response)]
    }

def create_protest_guidance_graph():
    """Creates the enhanced LangGraph workflow with parallel processing."""
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    # Web search nodes
    workflow.add_node("economic_web_search", economic_web_search_node)
    workflow.add_node("political_web_search", political_web_search_node)
    workflow.add_node("social_web_search", social_web_search_node)
    
    # RAG search nodes
    workflow.add_node("economic_rag_search", economic_rag_search_node)
    workflow.add_node("political_rag_search", political_rag_search_node)
    workflow.add_node("social_rag_search", social_rag_search_node)
    
    # Processing nodes
    workflow.add_node("merge_data", merge_data_node)
    workflow.add_node("economic_analysis", economic_analysis_node)
    workflow.add_node("political_analysis", political_analysis_node)
    workflow.add_node("social_analysis", social_analysis_node)
    workflow.add_node("safety_analysis", safety_analysis_node)
    workflow.add_node("legal_analysis", legal_analysis_node)
    workflow.add_node("final_synthesis", final_synthesis_node)
    
    # Set entry point - start with parallel web searches
    workflow.add_edge(START, "economic_web_search")
    workflow.add_edge(START, "political_web_search")
    workflow.add_edge(START, "social_web_search")
    
    # Also start RAG searches in parallel
    workflow.add_edge(START, "economic_rag_search")
    workflow.add_edge(START, "political_rag_search")
    workflow.add_edge(START, "social_rag_search")
    
    # All searches converge to merge_data
    workflow.add_edge("economic_web_search", "merge_data")
    workflow.add_edge("political_web_search", "merge_data")
    workflow.add_edge("social_web_search", "merge_data")
    workflow.add_edge("economic_rag_search", "merge_data")
    workflow.add_edge("political_rag_search", "merge_data")
    workflow.add_edge("social_rag_search", "merge_data")
    
    # After merging, start parallel analyses
    workflow.add_edge("merge_data", "economic_analysis")
    workflow.add_edge("merge_data", "political_analysis")
    workflow.add_edge("merge_data", "social_analysis")
    
    # All analyses feed into safety analysis
    workflow.add_edge("economic_analysis", "safety_analysis")
    workflow.add_edge("political_analysis", "safety_analysis")
    workflow.add_edge("social_analysis", "safety_analysis")
    
    # Safety leads to legal
    workflow.add_edge("safety_analysis", "legal_analysis")
    
    # Legal leads to final synthesis
    workflow.add_edge("legal_analysis", "final_synthesis")
    
    # End
    workflow.add_edge("final_synthesis", END)
    
    return workflow.compile()

# Initialize the graph
protest_graph = create_protest_guidance_graph()

def get_agent():
    """Returns the enhanced LangGraph-based protest guidance system."""
    print("Protest Guidance Agent Graph Structure:")
    try:
        print(protest_graph.get_graph().draw_ascii())
    except:
        print("Graph visualization not available")
    return protest_graph


# Helper function to run the agent
def run_protest_guidance(question: str):
    """Run the protest guidance agent with a question."""
    agent = get_agent()
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "web_results": "",
        "rag_results": "",
        "economic_web_data": "",
        "political_web_data": "",
        "social_web_data": "",
        "economic_rag_data": "",
        "political_rag_data": "",
        "social_rag_data": "",
        "economic_analysis": "",
        "political_analysis": "",
        "social_analysis": "",
        "safety_analysis": "",
        "legal_analysis": "",
        "final_answer": ""
    }
    
    result = agent.invoke(initial_state)
    return result["final_answer"]
