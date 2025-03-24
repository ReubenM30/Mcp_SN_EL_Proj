import os
import asyncio
import chainlit as cl
from typing import Literal
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY in .env!")

# Define MCP server path
MCP_SERVER_PATH = os.path.abspath("./elasticsearch-mcp-server/main.py")

# Initialize base LLM
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    openai_api_key=openai_api_key
)

# ---------------------- Tool Definitions ----------------------

@tool
def sonar_status(AppID: str) -> str:
    """Fetches the SonarQube analysis report for the given AppID."""
    return f"✅ SonarQube Analysis for AppID {AppID}: Passed with 2 critical issues."

@tool
def splunk_status(AppID: str) -> str:
    """Fetches the latest logs from Splunk for the given AppID."""
    return f"✅ Splunk Status for AppID {AppID}: No major errors found in the last 24 hours."

@tool
def appdynamics_status(AppID: str) -> str:
    """Fetches the latest application performance insights from AppDynamics for the given AppID."""
    return f"✅ AppDynamics Status for AppID {AppID}: Memory usage at 70%, CPU stable."

@tool
def overall_status(AppID: str) -> str:
    """Fetches an overall health report by aggregating results from SonarQube, Splunk, and AppDynamics."""
    sonar = sonar_status(AppID)
    splunk = splunk_status(AppID)
    appdynamics = appdynamics_status(AppID)
    return f"📊 **Overall Status for AppID {AppID}**:\n\n{sonar}\n{splunk}\n{appdynamics}"

@tool
def servicenow_tool(query: str) -> str :
    """
    Forwards any ServiceNow related natural language query to a locally running MCP tool server.
    """

    async def run_servicenow_agent():
        try:
            server_params = StdioServerParameters(
                command="C://Users//user//AppData//Local//Programs//Python//Python313//python.exe",
                args=[
                    "-m",
                    "mcp_server_servicenow.cli",
                    "--url", "https://dev268377.service-now.com/",
                    "--username", "admin",
                    "--password", "A72D^ksF$oFc"
                ],
            )
            
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)

                    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
                    agent = create_react_agent(model, tools)

                    result = await agent.ainvoke({"messages": query})
                    return str(result)

        except Exception as e:
            return f"❌ ServiceNow failed: {str(e)}"

    return asyncio.run(run_servicenow_agent())



@tool
def elastic_tool(query: str) -> str:
    """
    Forwards any ElasticSearch-related natural language query to a locally running MCP tool server.
    """

    async def run_elastic_agent():
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[MCP_SERVER_PATH]
            )
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await load_mcp_tools(session)

                    model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai_api_key)
                    agent = create_react_agent(model, tools)

                    result = await agent.ainvoke({"messages": query})
                    return str(result)

        except Exception as e:
            return f"❌ ElasticTool failed: {str(e)}"

    return asyncio.run(run_elastic_agent())

# ---------------------- Register Tools ----------------------

tools = [sonar_status, splunk_status, appdynamics_status, overall_status, elastic_tool, servicenow_tool]

llm = llm.bind_tools(tools).with_config(
    system_prompt=(
        "You are a monitoring assistant responsible for checking application health. "
        "You have access to the following tools:\n"
        "- SonarQube (`sonar_status`)\n"
        "- Splunk (`splunk_status`)\n"
        "- AppDynamics (`appdynamics_status`)\n"
        "- ElasticSearch (`elastic_tool`) - Accepts natural language queries for Elastic logs, metrics, or errors\n"
        "- ServiceNow (`servicenow_tool`) - Accepts natural language queries for servicenow incidents or issues\n"
        "- Overall Status (`overall_status`) - Calls all other tools at once\n\n"
        "Use `elastic_tool` when the user asks any ElasticSearch-related question like 'list indices', 'search logs', or 'error trends'."
        "All Elastic Operations that end user may ask are all supported by elastic_tool"
    )
)

# ---------------------- LangGraph Nodes ----------------------

tool_node = ToolNode(tools=tools)

def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        tool_names = [tc['name'] for tc in last_message.tool_calls]
        print(f"🛠️ Tool(s) invoked: {tool_names}")
        return "tools"
    return "final"

def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm.invoke(messages)
    if not response:
        print("🔴 [ERROR] LLM did not return a response!")
        response = AIMessage(content="⚠️ LLM failed to generate a response.")
    return {"messages": messages + [response]}

def pass_through_final(state: MessagesState):
    messages = state["messages"]
    if not messages or not messages[-1].content.strip():
        print("🔴 [ERROR] No final response generated!")
        return {"messages": [AIMessage(content="⚠️ No response generated. Please try again!")]}
    return {"messages": messages}

# ---------------------- Build Graph ----------------------

builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("final", pass_through_final)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "final")
builder.add_edge("final", END)

graph = builder.compile()

# ---------------------- Chainlit Integration ----------------------

def get_chat_history(new_message):
    # Initialize history if not already present
    if not hasattr(cl.user_session, "chat_history"):
        cl.user_session.chat_history = []
    cl.user_session.chat_history.append(new_message)
    return cl.user_session.chat_history


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")

    input_messages = get_chat_history(HumanMessage(content=msg.content))

    full_response = ""
    async for msg, metadata in graph.astream(
        {"messages": input_messages},
        stream_mode="messages",
        config=RunnableConfig(callbacks=[cb], **config),
    ):
        if msg.content.strip():
            print(f"🟡 [DEBUG] Streaming Step: {msg.content}")
            full_response += msg.content
            await final_answer.stream_token(msg.content)

    final_answer.content = full_response.strip() or "⚠️ No response generated! Please try again."
    cl.user_session["chat_history"].append(final_answer)
    await final_answer.send()
