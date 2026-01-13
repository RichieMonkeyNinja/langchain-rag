from langchain_ollama import ChatOllama
from langchain.messages import AIMessage
from langchain.tools import tool

import datetime 

@tool
def get_today_date():
    '''
    Get today's date
    '''
    return datetime.datetime.now().strftime("%Y-%m-%d")


llm = ChatOllama(
    model='qwen3:4b',
    temperature=0,
    reasoning=False
)

messages = [
    ("human", "What is today's date?")
]

# streamer = llm.stream(messages)
# print(streamer)

for chunk in llm.stream(messages):
    if chunk.content:
        print(chunk.content, end = '', flush=True)
    
    if chunk.tool_call_chunks:
        for tc_chunk in chunk.tool_call_chunks:
            if tc_chunk.get("name"):
                print(f'LLM is calling tool: {tc_chunk['name']}')

ai_msg = llm.invoke(messages)
print(ai_msg)

if isinstance(ai_msg, AIMessage) and ai_msg.tool_calls:
    print(ai_msg.tool_calls)