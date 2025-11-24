# streamlit_app.py

import os
import uuid
import operator
import sqlite3
from typing import Annotated, List, Dict, TypedDict
from dotenv import load_dotenv

import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# -----------------------------
# API KEYS
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


if not OPENAI_API_KEY:
    st.warning("Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")
else:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not TAVILY_API_KEY:
    st.warning("Please set TAVILY_API_KEY in Streamlit secrets or environment variables.")
else:
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# -----------------------------
# LLM & Tools
# -----------------------------
llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
tavily_tool = TavilySearchResults(max_results=3)
tools = [tavily_tool]

# SQLite in-memory for checkpointing
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)

# -----------------------------
# State
# -----------------------------
def merge_sections(a: Dict, b: Dict) -> Dict:
    return {**a, **b}

class AccountPlanState(TypedDict):
    task: str
    sections: Annotated[Dict[str, str], merge_sections]
    section_status: Annotated[Dict[str, str], merge_sections]
    warnings: Annotated[List[str], operator.add]
    messages: Annotated[List[BaseMessage], operator.add]
    draft_plan: str

# -----------------------------
# Prompts
# -----------------------------
financial_prompt = """You are a Financial Analyst. Research the target company.
Focus on: Revenue, Net Income, and Debt.
IMPORTANT: If you find ANY discrepancy in numbers (even small ones), start your response with "CONFLICT_DETECTED:" followed by the details.
Otherwise, provide a clean summary."""

market_prompt = """You are a Market Strategist. Research the target company.
Focus on: Market Share and Competitors.
IMPORTANT: If you find conflicting reports on market position, start your response with "CONFLICT_DETECTED:" followed by the details.
Otherwise, provide a clean summary."""

people_prompt = """You are a Leadership Scout. Research the target company.
Focus on: CEO, C-Suite, and recent layoffs.
IMPORTANT: If you find mixed news about leadership changes, start your response with "CONFLICT_DETECTED:" followed by the details.
Otherwise, provide a clean summary."""

writer_prompt = """You are a Strategy Consultant.
Compile the following research into a Professional Account Plan in Markdown.
Research:
{research_data}

Format:
- Executive Summary
- Financial Overview
- Market Position
- Leadership & Culture
- Strategic Recommendations
"""

from langchain.agents import create_agent

# -----------------------------
# Node functions
# -----------------------------
def financial_node(state: AccountPlanState):
    agent = create_agent(llm, tools, system_prompt=financial_prompt)
    result = agent.invoke({"messages": [{"role": "user", "content": f"Research financials for: {state['task']}"}]})
    content = result["messages"][-1].content
    warn = [f"Financials: {content[:5000]}..."] if "CONFLICT_DETECTED" in content else []
    return {"sections": {"financials": content}, "section_status": {"financials": "done"}, "warnings": warn}

def market_node(state: AccountPlanState):
    agent = create_agent(llm, tools, system_prompt=market_prompt)
    result = agent.invoke({"messages": [{"role": "user", "content": f"Research market strategy for: {state['task']}"}]})
    content = result["messages"][-1].content
    warn = [f"Market: {content[:5000]}..."] if "CONFLICT_DETECTED" in content else []
    return {"sections": {"competitors": content}, "section_status": {"competitors": "done"}, "warnings": warn}

def people_node(state: AccountPlanState):
    agent = create_agent(llm, tools, system_prompt=people_prompt)
    result = agent.invoke({"messages": [{"role": "user", "content": f"Research leadership for: {state['task']}"}]})
    content = result["messages"][-1].content
    warn = [f"Leadership: {content[:5000]}..."] if "CONFLICT_DETECTED" in content else []
    return {"sections": {"leadership": content}, "section_status": {"leadership": "done"}, "warnings": warn}

def writer_node(state: AccountPlanState):
    sections = state.get("sections", {})
    combined_text = (
        f"Financials: {sections.get('financials')}\n"
        f"Market: {sections.get('competitors')}\n"
        f"Leadership: {sections.get('leadership')}"
    )
    response = llm.invoke([
        SystemMessage(content=writer_prompt.format(research_data=combined_text)),
        HumanMessage(content=f"Write the plan for {state['task']}")
    ])
    return {"draft_plan": response.content}

def supervisor_router(state: AccountPlanState):
    status = state.get("section_status", {})
    to_run = []
    if status.get("financials") != "done":
        to_run.append("financial_node")
    if status.get("competitors") != "done":
        to_run.append("market_node")
    if status.get("leadership") != "done":
        to_run.append("people_node")
    if not to_run:
        return ["join_node"]
    return to_run

def join_node(state: AccountPlanState):
    return {}

def human_review_node(state: AccountPlanState):
    return {}

def review_gate(state: AccountPlanState):
    if state.get("warnings") and state.get("section_status", {}).get("review") != "done":
        return "human_review_node"
    return "writer_node"

# -----------------------------
# Workflow
# -----------------------------
workflow = StateGraph(AccountPlanState)
workflow.add_node("supervisor", lambda s: {})
workflow.add_node("financial_node", financial_node)
workflow.add_node("market_node", market_node)
workflow.add_node("people_node", people_node)
workflow.add_node("join_node", join_node)
workflow.add_node("human_review_node", human_review_node)
workflow.add_node("writer_node", writer_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    supervisor_router,
    ["financial_node", "market_node", "people_node", "join_node"],
)
workflow.add_edge("financial_node", "join_node")
workflow.add_edge("market_node", "join_node")
workflow.add_edge("people_node", "join_node")
workflow.add_conditional_edges(
    "join_node",
    review_gate,
    {"human_review_node": "human_review_node", "writer_node": "writer_node"},
)
workflow.add_edge("human_review_node", "writer_node")
workflow.add_edge("writer_node", END)
app = workflow.compile(checkpointer=memory, interrupt_before=["human_review_node"])

# -----------------------------
# Session State
# -----------------------------
if "config" not in st.session_state:
    st.session_state.config = None
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "paused_for_review" not in st.session_state:
    st.session_state.paused_for_review = False
if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "company_name" not in st.session_state:
    st.session_state.company_name = ""

# -----------------------------
# User type detection
# -----------------------------
def detect_user_type(user_input: str) -> str:
    input_text = user_input.strip()
    input_lower = input_text.lower()
    if not input_text:
        return "Confused"

    confused_keywords = ["not sure", "maybe", "i think", "unsure", "could be", "possibly"]
    if any(keyword in input_lower for keyword in confused_keywords):
        return "Confused"

    if (input_text.isnumeric() or
        len(input_text) > 250 or  
        not any(c.isalpha() for c in input_text) or
        all(not c.isalnum() for c in input_text)):
        return "Edge Case"

    chatty_phrases = ["also", "by the way", "oh", "just wondering", "btw", "i was thinking", "as well"]
    if len(input_text.split()) > 25 or any(phrase in input_lower for phrase in chatty_phrases):
        return "Chatty"

    return "Efficient"


def guidance_for_user_type(user_input: str) -> str:
    user_type = detect_user_type(user_input)
    print(user_type)
    if user_type == "Confused":
        return "User seems unsure. Prioritize verified sources and clarify assumptions."
    elif user_type == "Efficient":
        return "User expects quick results. Summarize key insights concisely."
    elif user_type == "Chatty":
        return "User provides long input. Extract core points and ignore off-topic content."
    elif user_type == "Edge Case":
        return "Input may be invalid or ambiguous. Handle with safe defaults and robust error checking."
    return "Proceed normally."

# -----------------------------
# Main functions
# -----------------------------
def start_new_run(company_name: str):
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    st.session_state.config = config
    st.session_state.company_name = company_name
    st.session_state.warnings = []
    st.session_state.paused_for_review = False
    st.session_state.final_report = None

    initial_input = {
        "task": company_name,
        "sections": {},
        "section_status": {},
        "warnings": [],
    }

    guidance = guidance_for_user_type(company_name)

    with st.spinner(f"Parallel researching: {company_name}..."):
        for event in app.stream(initial_input, config):
            for key in event:
                if key not in ["__start__", "__end__"]:
                    st.write(f"Finished: `{key}`")

    snapshot = app.get_state(config)
    if not snapshot.next:
        final_report = snapshot.values.get("draft_plan")
        st.session_state.final_report = final_report
        st.session_state.paused_for_review = False
    else:
        if snapshot.next[0] == "human_review_node":
            warnings = snapshot.values.get("warnings", [])
            st.session_state.warnings = warnings
            st.session_state.paused_for_review = True

def resume_after_guidance(guidance: str):
    config = st.session_state.config
    if config is None:
        st.error("No active run found. Please start again.")
        return
    if not guidance:
        guidance = "Proceed with the data you have."

    snapshot = app.get_state(config)
    current_values = snapshot.values or {}
    current_task = current_values.get("task", st.session_state.company_name)
    existing_section_status = current_values.get("section_status", {}) or {}
    merged_section_status = {
        **existing_section_status,
        "financials": existing_section_status.get("financials", "done"),
        "competitors": existing_section_status.get("competitors", "done"),
        "leadership": existing_section_status.get("leadership", "done"),
        "review": "done",
    }
    existing_messages = current_values.get("messages", []) or []
    new_messages = existing_messages + [HumanMessage(content=f"User Guidance: {guidance}")]
    app.update_state(config, {
        "task": current_task,
        "section_status": merged_section_status,
        "messages": new_messages,
    })

    with st.spinner("Resuming execution with guidance..."):
        for event in app.stream(None, config):
            for key in event:
                if key not in ["__start__", "__end__"]:
                    st.write(f"Finished: `{key}`")

    snapshot = app.get_state(config)
    final_report = snapshot.values.get("draft_plan")
    st.session_state.final_report = final_report
    st.session_state.paused_for_review = False

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Account Plan Agent")
st.caption("Researches financials, market, and leadership in parallel and generates a professional account plan.")

company_input = st.text_input("Please enter the query.", value=st.session_state.company_name or "Oracle Corporation")

if st.button("Run / Restart Research"):
    if not OPENAI_API_KEY or not TAVILY_API_KEY:
        st.error("Please configure your API keys first.")
    elif not company_input.strip():
        st.error("Please enter the query.")
    else:
        start_new_run(company_input.strip())

if st.session_state.paused_for_review:
    st.markdown("Paused: Human Review Required")
    warnings = st.session_state.warnings or []
    if warnings:
        st.warning(f"Found {len(warnings)} potential conflicts.")
        with st.expander("View conflict details"):
            for i, w in enumerate(warnings, start=1):
                st.markdown(f"**{i}.** {w}")
    else:
        st.info("No explicit conflicts were captured, but the workflow paused for review.")

    guidance_text = st.text_area(
        "Your guidance to the agent",
        placeholder="E.g., 'Trust the latest annual report over news articles.'",
        height=120
    )

    if st.button("Resume with this guidance"):
        resume_after_guidance(guidance_text)

if st.session_state.final_report:
    st.markdown("## Final Account Plan")
    st.markdown(st.session_state.final_report)
