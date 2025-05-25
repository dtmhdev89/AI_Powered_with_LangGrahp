from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=4) #increased number of results
print(type(tool))
print(tool.name)

# <class 'langchain_community.tools.tavily_search.tool.TavilySearchResults'>
# tavily_search_results_json
# LLms will use the name of the tool to call it


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:

    def __init__(self, model, tools, system=""):
        self.system = system
        # create the graph by initializing the state graph of the agent
        # This graph doesn't have any node added to it at the initialization
        graph = StateGraph(AgentState)
        # As the design, adding the llm node, with the function to call as representation for llms
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)  
        graph.add_conditional_edges(
            "llm",  # node where the edge start
            self.exists_action,  # the function that determine where to go after that
            {True: "action", False: END} # dictionary representing how to map the response of the function to the next node to go to. True will go to action node, False will go to END node
        )
        # regular edge will go from the action node to the LLM node
        graph.add_edge("action", "llm")
        # entrypoing for the graph
        graph.set_entry_point("llm")
        # compile the graph. This should be called after having done all the setups.
        # This will turn the graph into a LangChain runnable
        # A LangChain runnable exposes a standard interface for calling and invoking graph
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        # pass list of tools into the agent
        # it let model know that it has these tools available to call
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        # should be return with boolean on conditional edge
        result = state['messages'][-1]
        # check if any tool_calls, True if any tool_calls
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        # get list of messages from the AgentState
        messages = state['messages']
        if self.system:
            # Add the system message into the beginning of messages from state
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        # return the message from the model
        # this in dictionary of {'messages': [...]]} to update the state of messages in AgentSate by adding it
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        # get the last message from the list of messages in the state and get the attribute tool_calls
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:      # check for bad tool name from LLM
                print("\n ....bad tool name....")
                result = "bad tool name, retry"  # instruct LLM to retry if bad
            else:
                # invoke tool calling
                result = self.tools[t['name']].invoke(t['args'])
            # Add the ToolMessage into the results
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        # this in dictionary of {'messages': [...]]} to update the state of results in AgentSate by adding it
        return {'messages': results}

# Initialize the the prompt and agent
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatOpenAI(model="gpt-3.5-turbo")  #reduce inference cost
abot = Agent(model, [tool], system=prompt)

# Visualize the graph created
from IPython.display import Image

Image(abot.graph.get_graph().draw_png())

# HumanMessage to represent the user message
messages = [HumanMessage(content="What is the weather in sf?")]
# we have to pass {"messages": messages} since the state that the agent expects to work with
# has this messages attribute, which is a list of messages
# we need to make it conform with that state
result = abot.graph.invoke({"messages": messages})
# # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX'}
# Back to the model!
# # result is the final state that the agent ended up in.

print(result)
# # {'messages': [HumanMessage(content='What is the weather in sf?'),
# # AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX', 'function': {'arguments': '{"query":"weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 153, 'total_tokens': 174, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-421653e5-2deb-4bdd-95dc-9f7db7f844a7-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_PvPN1v7bHUxOdyn4J2xJhYOX'}]),
# # ToolMessage(content='[{\'url\': \'https://weathershogun.com/weather/usa/ca/san-francisco/480/may/2025-05-23\', \'content\': \'San Francisco, California Weather: Friday, May 23, 2025. Cloudy weather, overcast skies with clouds. Day 68°. Night 54°. Precipitation 0 %.\'}, {\'url\': \'https://www.easeweather.com/north-america/united-states/california/city-and-county-of-san-francisco/san-francisco/may\', \'content\': \'Sunny\\n| 66° /50° | 0\\xa0in | 5 |  |\\n| May 22 | \\nSunny\\n| 66° /48° | 0\\xa0in | 6 |  |\\n| May 23 | \\nSunny\\n| 64° /50° | 0\\xa0in | 5 |  |\\n| May 24 | \\nSunny\\n| 68° /50° | 0\\xa0in | 5 |  |\\n| May 25 | \\nPartly cloudy\\n| 66° /50° | 0\\xa0in | 5 |  |\\n| May 26 | \\nSunny\\n| 62° /50° | 0\\xa0in | 5 |  |\\n| May 27 | \\nSunny\\n| 64° /50° | 0\\xa0in | 5 |  |\\n| May 28 | \\nPatchy rain possible\\n| 62° /50° | 0\\xa0in | 4 |  |\\n| May 29 | \\nCloudy\\n| 64° /48° | 0\\xa0in | 5 |  |\\n| May 30 | \\nSunny\\n| 68° /50° | 0\\xa0in | 6 |  |\\n| May 31 | \\nSunny [...] More\\n\\nNew! Chat with our AI weatherman - it’s amazing and free.\\n\\n\\n28\\nDry days\\n\\n3\\nRainy days\\n\\n0\\nSnow days\\n\\n64.4°/50°\\nTemperatures\\n\\n12.4\\xa0mph\\nAvg max wind\\n\\n72 %\\nAvg humidity [...] Enjoy moderate daytime temperatures of up to 64° and cooler nights around 50°. Ideal for a range of outdoor activities, from hiking to city tours.\\nHow much rainfall should be expected in San Francisco during May?\\nExpect minimal rainfall in San Francisco during May, with a total of 0.47\\xa0in over approximately 3 days. This minimal precipitation is unlikely to significantly impact plans.\\nWhat are the typical wind conditions in San Francisco during May?\'}, {\'url\': \'https://world-weather.info/forecast/usa/san_francisco/may-2025/\', \'content\': "Weather in San Francisco in May 2025\\n\\nSan Francisco Weather Forecast for May 2025 is based on long term prognosis and previous years\' statistical data.\\n\\nMay\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\n+50°\\n\\n+57°\\n\\n+55°\\n\\n+54°\\n\\n+50°\\n\\n+52°\\n\\n+54°\\n\\n+52°\\n\\n+55°\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\n+54°\\n\\n+52°\\n\\n+54°\\n\\n+52°\\n\\n+50°\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\n+54°\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\n+52°\\n\\nExtended weather forecast in San Francisco\\n\\nWeather in large and nearby cities\\n\\nWeather in Washington, D.C.+57°\\n\\nSacramento+66°\\n\\nPleasanton+55°"}, {\'url\': \'https://weatherspark.com/h/m/557/2025/5/Historical-Weather-in-May-2025-in-San-Francisco-California-United-States\', \'content\': \'as good as the data that underpin them, that weather conditions at any given location and time are unpredictable and variable, and that the definition of the scores reflects a particular set of preferences that may not agree with those of any particular reader.Please review our full terms contained on ourTerms of Servicepage. | 6-Hour Low | 64.0Â°F | 6-Hour High | 78.1Â°F | Dew Pt. | 48.0Â°Fdry | Rel. Humidity | 44% | Wind Dir. | 300 deg, WNW | Vis. | 10.00 mi or greater | Alt. | 29.99 inHg | [...] Low64.0Â°F6-Hour High78.1Â°FDew Pt.48.0Â°FdryRel. Humidity44%PrecipitationNo ReportWind18.4mphfresh breezeWind Dir.300 deg, WNWCloud CoverMostly Clear20,000 ftVis.10.00 mi or greaterAlt.29.99 inHgRaw: KSFO 092356Z 30016KT 10SM FEW200 22/09 A2999 RMK AO2 SLP154 T02170089 10256 20178 56016 $This report shows the past weather for San Francisco, providing a weather history for May 2025. It features all historical weather data series we have available, including the San Francisco temperature history [...] PMS248,577 mi783%-3:45 AMW3:45 PME9:57 PMS250,419 mi890%-4:06 AMW4:43 PME10:36 PMS251,642 mi995%-4:26 AMW5:41 PMESE11:15 PMS252,288 mi1098%-4:48 AMWSW6:40 PMESE11:57 PMS252,412 mi1199%-5:13 AMWSW7:40 PMESE--12100%-5:41 AMWSW8:42 PMESE12:41 AMS252,068 mi13100%-6:14 AMWSW9:43 PMSE1:28 AMS251,297 mi1497%-6:55 AMSW10:42 PMSE2:19 AMS250,121 mi1593%-7:43 AMSW11:36 PMSE3:13 AMS248,547 mi1687%-8:40 AMSW-4:08 AMS246,573 mi1780%12:23 AMSE9:42 AMWSW-5:02 AMS244,198 mi1871%1:03 AMESE10:48 AMWSW-5:55\'}]', name='tavily_search_results_json', tool_call_id='call_PvPN1v7bHUxOdyn4J2xJhYOX'),
# # AIMessage(content='The weather in San Francisco today is cloudy with overcast skies and temperatures reaching 68°F during the day and 54°F at night. There is no precipitation expected.', response_metadata={'token_usage': {'completion_tokens': 35, 'prompt_tokens': 1689, 'total_tokens': 1724, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run-ffa5c8b1-a24d-4352-a519-e10acdf7fff7-0')]}

print(result['messages'][-1].content)
# 'The weather in San Francisco today is cloudy with overcast skies and temperatures 
# reaching 68°F during the day and 54°F at night. There is no precipitation expected.'

messages = [HumanMessage(content="What is the weather in SF and LA?")]
result = abot.graph.invoke({"messages": messages})
# # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in San Francisco'}, 'id': 'call_1SqGYuEtOOFN1yiIHSQTPnvE'}
# # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'weather in Los Angeles'}, 'id': 'call_8RiM72Y7G8V7c3HEEAML1SKP'}
# # Back to the model!
# # This considers an example of parallel function or tool calling

print(result['messages'][-1].content)

# # 'The weather in San Francisco on May 23, 2025, is cloudy with overcast skies. 
# # The daytime temperature is 68°F and the nighttime temperature is 54°F with 0% precipitation.
# # \n\nIn Los Angeles on May 23, 2025, the weather is also cloudy with overcast skies.
# # The daytime temperature is 77°F and the nighttime temperature is 59°F with 0% precipitation.'

# More complex query
query = "Who won the super bowl in 2024? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question." 
messages = [HumanMessage(content=query)]

model = ChatOpenAI(model="gpt-4o")  # requires more advanced model
abot = Agent(model, [tool], system=prompt)
result = abot.graph.invoke({"messages": messages})
# # Calling: {'name': 'tavily_search_results_json', 'args': {'query': '2024 Super Bowl winner'}, 'id': 'call_HBUU1Lo9WSgKCPKYCAStSb7g'}
# # Back to the model!
# # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Kansas City Chiefs headquarters state'}, 'id': 'call_p79advdYztGa70UzKW2wPL2C'}
# # Back to the model!
# # Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'Missouri GDP 2024'}, 'id': 'call_X73Bco04TXOLFYVIgOJBD5Ss'}
# # Back to the model!

# # It called the action sequentially because the later question needed the response from previous question

print(result['messages'][-1].content)

# 1. The winner of the 2024 Super Bowl (Super Bowl LVIII) was the Kansas City Chiefs, who defeated the San Francisco 49ers 25-22 in overtime.

# 2. The headquarters of the Kansas City Chiefs is located in Kansas City, Missouri.

# 3. As of the end of 2024, Missouri's GDP was approximately $460.7 billion.
