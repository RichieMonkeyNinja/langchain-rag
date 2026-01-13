from langchain.messages import HumanMessage
from langchain_core.messages import ChatMessage
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = 'gemma3:4b'
)

messages = [
    ChatMessage(role='control',content='thinking'),
    HumanMessage('Is 69^5 = 1564031349')
]

for chunk in llm.stream(messages):
    if chunk.content:
        print(chunk.content, end = '', flush=True)
    
    if chunk.tool_call_chunks:
        for tc_chunk in chunk.tool_call_chunks:
            if tc_chunk.get("name"):
                print(f'LLM is calling tool: {tc_chunk['name']}')

