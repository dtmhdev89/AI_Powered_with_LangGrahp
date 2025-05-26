# ReAct Pattern
import openai
import re
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()
from openai import OpenAI

client = OpenAI()

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
)

chat_completion.choices[0].message.content


class Agent:
    def __init__(self, system=""):
        # want this Agent to be parameterized by a system message
        self.system = system
        # keep track of a list of messages over time. Append eveything happened in ReAct loop
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        # append the result after executing a function
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        # call the LLMs
        completion = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,  # make the LLMs response very deterministic
                        messages=self.messages)  # pass the accumulated messages
        return completion.choices[0].message.content
    

# To create ReAct agent, requiring a very specific system message
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

average_dog_weight:
e.g. average_dog_weight: Collie
returns average weight of a dog when given the breed

Example session:

Question: How much does a Bulldog weigh?
Thought: I should look the dogs weight using average_dog_weight
Action: average_dog_weight: Bulldog
PAUSE

You will be called again with this:

Observation: A Bulldog weights 51 lbs

You then output:

Answer: A bulldog weights 51 lbs
""".strip()


# Define the functions in the prompt
def calculate(what):
    return eval(what)

def average_dog_weight(name):
    if name in "Scottish Terrier": 
        return("Scottish Terriers average 20 lbs")
    elif name in "Border Collie":
        return("a Border Collies average weight is 37 lbs")
    elif name in "Toy Poodle":
        return("a toy poodles average weight is 7 lbs")
    else:
        return("An average dog weights 50 lbs")

known_actions = {
    "calculate": calculate,
    "average_dog_weight": average_dog_weight
}

# Try it out
abot = Agent(prompt)
result = abot("How much does a toy poodle weight?")
print(result)

# # result
# # # Thought: I should look up the average weight of a Toy Poodle using the average_dog_weight action.
# # # Action: average_dog_weight: Toy Poodle
# # # PAUSE

# It wait for an Observation
result = average_dog_weight("Toy Poodle")
# # result
# # # 'a toy poodles average weight is 7 lbs'

# Pass the observation result into prompt
next_prompt = "Observation: {}".format(result)

# then call the agent with next_prompt
abot(next_prompt)
# # Answer will be generated
# # 'Answer: A Toy Poodle weighs an average of 7 lbs.'

# To see more detail about what exactly has been going on
abot.messages

# Another try out
abot = Agent(prompt)
question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
abot(question)
# # Result
# # # 'Thought: I need to find the average weight of both a Border Collie and a Scottish Terrier, 
# # # then add them together to find the combined weight.\nAction: average_dog_weight: Border Collie\nPAUSE'
next_prompt = "Observation: {}".format(average_dog_weight("Border Collie"))
print(next_prompt)
# # Observation: a Border Collies average weight is 37 lbs
abot(next_prompt)
# # 'Action: average_dog_weight: Scottish Terrier\nPAUSE'
next_prompt = "Observation: {}".format(average_dog_weight("Scottish Terrier"))
print(next_prompt)
# # Observation: Scottish Terriers average 20 lbs
abot(next_prompt)
# # 'Thought: Now that I have the average weights of both dogs, I can calculate their combined weight 
# # by adding the two values together.\nAction: calculate: 37 + 20\nPAUSE'
next_prompt = "Observation: {}".format(eval("37 + 20"))
print(next_prompt)
# # Observation: 57
abot(next_prompt)
# # 'Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.'

# Loop way

# First build a regex
# python regular expression to selection action
action_re = re.compile('^Action: (\w+): (.*)$')


# Query method to run the same method as running manually
def query(question, max_turns=5):
    # keep track of how many iterations we've done
    i = 0
    bot = Agent(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print(result)

        # parse the action from response
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            # There is an action to run
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception("Unknown action: {}: {}".format(action, action_input))
            print(" -- running {} {}".format(action, action_input))
            observation = known_actions[action](action_input)
            print("Observation:", observation)
            next_prompt = "Observation: {}".format(observation)
        else:
            return


question = """I have 2 dogs, a border collie and a scottish terrier. \
What is their combined weight"""
query(question)

# Thought: I need to find the average weight of both a Border Collie and a Scottish Terrier, then add them together to find the combined weight.
# Action: average_dog_weight: Border Collie
# PAUSE
#  -- running average_dog_weight Border Collie
# Observation: a Border Collies average weight is 37 lbs
# Action: average_dog_weight: Scottish Terrier
# PAUSE
#  -- running average_dog_weight Scottish Terrier
# Observation: Scottish Terriers average 20 lbs
# Thought: Now that I have the average weights of both dogs, I can calculate their combined weight by adding the two values together.
# Action: calculate: 37 + 20
# PAUSE
#  -- running calculate 37 + 20
# Observation: 57
# Answer: The combined weight of a Border Collie and a Scottish Terrier is 57 lbs.
