from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

from uuid import uuid4
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage

"""
In previous examples we've annotated the `messages` state key
with the default `operator.add` or `+` reducer, which always
appends new messages to the end of the existing messages array.

Now, to support replacing existing messages, we annotate the
`messages` key with a customer reducer function, which replaces
messages with the same `id`, and appends them otherwise.
"""
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged


# For human-in-the-loop interactions, we may want to replace existing messages
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


tool = TavilySearchResults(max_results=2)


class Agent:
    def __init__(self, model, tools, system="", checkpointer=None):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]  # it's going to add an interrupt before we call the "action" node.
            # The reason is we're going to add somthing that requires manual approval
        )
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        print(state)
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


prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-3.5-turbo")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)


messages = [HumanMessage(content="Whats the weather in SF?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

# # {'messages': [HumanMessage(content='Whats the weather in SF?', id='3c687336-7a33-4c55-99ac-cf377af44ea3'), AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0', 'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 152, 'total_tokens': 174, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-1798a61c-f194-4ec5-a2e2-df833d5fd4e1-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0'}])]}
# # {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0', 'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 152, 'total_tokens': 174, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-1798a61c-f194-4ec5-a2e2-df833d5fd4e1-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0'}])]}
# # It stopped at AIMessage


# Get the current state of the graph of the thread with thread_id = 1 
abot.graph.get_state(thread)

# # StateSnapshot(values={'messages': [HumanMessage(content='Whats the weather in SF?', id='3c687336-7a33-4c55-99ac-cf377af44ea3'), AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-1798a61c-f194-4ec5-a2e2-df833d5fd4e1-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0'}])]}, next=('action',), config={'configurable': {'thread_id': '1', 'thread_ts': '1f037f60-2ea6-6614-8001-a163f8d11d64'}}, metadata={'source': 'loop', 'step': 1, 'writes': {'llm': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'logprobs': None, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'token_usage': {'completion_tokens': 22, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 152, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 174}}, id='run-1798a61c-f194-4ec5-a2e2-df833d5fd4e1-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_i7rGhnzgZf5hW3bsDcqdTrH0'}])]}}}, created_at='2025-05-23T16:50:19.888363+00:00', parent_config={'configurable': {'thread_id': '1', 'thread_ts': '1f037f60-2e50-6e22-8000-7c9a2ca122ba'}})

# To see the next node to be called
abot.graph.get_state(thread).next
# # ('action',)
# # This means we're about to call 'action' node

# to continue after interrupted
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)
# # it continues doing the flow and return the final response

# # Note that there was no break in between the "action" node and the "LLM" node
# # Because we didn't add any interrupt there.

abot.graph.get_state(thread)


abot.graph.get_state(thread).next

# # Nothing left to be done.

# Another
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)


# Modify State

messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "3"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)

abot.graph.get_state(thread)

# # Get current state
current_values = abot.graph.get_state(thread)

# # View last message in the current state
current_values.values['messages'][-1]

# # See the list of tool calls associated with this messages
current_values.values['messages'][-1].tool_calls
# # # [{'name': 'tavily_search_results_json',
# # #  'args': {'query': 'weather in Los Angeles'},
# # # 'id': 'call_6ED1ZQ8nrjYIOY14yqInLPZc'}]

# # Update these tool calls
_id = current_values.values['messages'][-1].tool_calls[0]['id']
current_values.values['messages'][-1].tool_calls = [
    {
        'name': 'tavily_search_results_json',
        'args': {'query': 'current weather in Louisiana'},
        'id': _id
    }
]

# # The update tool calls doesn't do anything until calling update on the state with the thread
# # together with the values want to update
abot.graph.update_state(thread, current_values.values)

# # get the current state
abot.graph.get_state(thread)

# # Continue the graph
for event in abot.graph.stream(None, thread):
    for v in event.values():
        print(v)


# Time Travel
states = []
for state in abot.graph.get_state_history(thread):
    print(state)
    print('--')
    states.append(state)

# get the last state
to_replay = states[-1]

# # StateSnapshot(values={'messages': [HumanMessage(content='Whats the weath..........

# # Resume at a specific state
# # In this case, need to pass the state's config (to_replay.config)
# # This means to_replay.config is the state where we want to resume
for event in abot.graph.stream(None, to_replay.config):
    for k, v in event.items():
        print(v)

# Go back in time and edit

# # Get the id of tool_calls and update the tool calls values
_id = to_replay.values['messages'][-1].tool_calls[0]['id']
to_replay.values['messages'][-1].tool_calls = [{'name': 'tavily_search_results_json',
  'args': {'query': 'current weather in LA, accuweather'},
  'id': _id}]

# # print out the to_replay.config
print(to_replay.config)
# # {'configurable': {'thread_id': '2',
# #  'thread_ts': '1f038bcd-5101-6bb0-8001-11e597982d06'}}

# # Calling update on the status to update the state
branch_state = abot.graph.update_state(to_replay.config, to_replay.values)

# # continue from the branch state where we want to resume
for event in abot.graph.stream(None, branch_state):
    for k, v in event.items():
        if k != "__end__":
            print(v)


# Add message to a state at given time

# # Get the _id of last message of the supposed-to-making tool call
_id = to_replay.values['messages'][-1].tool_calls[0]['id']

# # Propose that we don't want to call the Tavali tool, we want to mock out the response
# # by adding a new message into the sate
state_update = {"messages": [ToolMessage(
    tool_call_id=_id,
    name="tavily_search_results_json",
    content="54 degree celcius",
)]}

# # Update state of the graph that attached to to_replay.config with new values
# # The current state of the graph is about to go the "action" node.
# # But after, adding this message, we don't want it to go to the "action" node any more.
# # We update the state and we're acting as if we were the "action" node.
# # To act as the action node, we use as_node with the name of "action" node
branch_and_add = abot.graph.update_state(
    to_replay.config,
    state_update,
    as_node="action")

# # Continue from updated resume state
for event in abot.graph.stream(None, branch_and_add):
    for k, v in event.items():
        print(v)
