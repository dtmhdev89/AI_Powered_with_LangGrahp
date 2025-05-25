from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")

        # Add checkpointer to graph
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}


# To add persistence into the agent
from langgraph.checkpoint.sqlite import SqliteSaver

# SqliteSaver is a really simple check pointer that use sqlite under the hood (in-memory database in the example)
# can use other check pointer like redis, ...
memory = SqliteSaver.from_conn_string(":memory:")

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

# Add concept of Streaming
# Two things care about Streaming: streaming individual message (like AIMessage, Observation), tokens
# each tokens of the LLMs call we might want to stream the output
messages = [HumanMessage(content="What is the weather in sf?")]

# used to keep track of different threads inside the persistence check pointer
# this will allow us to have multiple conversations going on at the same time
# This is really needed for production application when generally having many users
thread = {"configurable": {"thread_id": "1"}}

# result is a stream of events
# these events represent updates to that state over time
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])

# continuous with the previous question. It's asking a follow-up question
# Don't tell explicitly about the weather, but based on it being a conversation.
# It expects it to realize that we're asking about the weather in the prompt
messages = [HumanMessage(content="What about in la?")]
# to make sure that we're continouing from the same point.
# Passing in the same thread id
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# # it knew we asked about the weather in LA. Because it has this persistence from the checkpointer

# Do the same as previous again
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# # it did the same thing

# change the thread id
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# # It behaves diferently. Since it doesn't have access to any history

# How to stream events
# Stream Token
from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

# Async stream event is an asynchronous method which means that 
# we're going to need to use an async checkpointer
memory = AsyncSqliteSaver.from_conn_string(":memory:")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="What is the weather in SF?")]
thread = {"configurable": {"thread_id": "4"}}
# looping through different events.
# These events represent updates from the underlying stream
async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
    kind = event["event"]
    # Looking for an event that correspond to new tokens
    # These kind of events are call on_chat_model_stream
    # When seeing this kind of event, should print it out
    # It will streaming real time into the screen
    if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
            # Empty content in the context of OpenAI means
            # that the model is asking for a tool to be invoked.
            # So we only print non-empty content
            print(content, end="|")

# Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_8etfZasfe0eUw99Q9qqzCQ0N'}
# Back to the model!
# The| current| weather| in| San| Francisco|,| California|,| is| cloudy| with| over|cast| skies|.
# | The| temperature| is| around| |68|°F| during| the| day| and| |54|°F| at| night|,| with| no| precipitation| expected|.|
