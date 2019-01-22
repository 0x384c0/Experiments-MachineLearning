import IPython
from IPython.display import clear_output, HTML, display
from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

print(bcolors.WARNING + "Loading..." + bcolors.ENDC)

interpreter = RasaNLUInterpreter('models/current/nlu')
messages = ["Hi! you can chat in this window. Type 'stop' to end the conversation."]
agent = Agent.load('models/dialogue', interpreter=interpreter)

def chatlogs_html(messages):
    messages_html = "".join(["<p>{}</p>".format(m) for m in messages])
    chatbot_html = """<div class="chat-window" {}</div>""".format(messages_html)
    return chatbot_html

print(bcolors.WARNING + "Loaded" + bcolors.ENDC)
while True:
    clear_output()
    print(bcolors.HEADER + "Bot response:" + bcolors.ENDC)
    print("    " + messages[-1])
    print(bcolors.HEADER + "Your message:" + bcolors.ENDC)
    time.sleep(0.3)
    a = input()
    messages.append(a)
    if a == 'stop' or a == 'q':
        break
    responses = agent.handle_message(a)
    for r in responses:
        messages.append(r.get("text"))