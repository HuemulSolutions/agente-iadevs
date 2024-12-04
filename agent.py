from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import ChatPromptTemplate
from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from tools import tools
import datetime, json
from dotenv import load_dotenv

import os
load_dotenv()

model = "gpt4-0"
temperature = 0
llm = AzureChatOpenAI(
    model=model,
    temperature=temperature,
    azure_deployment=os.getenv("GPT4O_DEPLOY"))

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(name="gpt4-0", temperature=0)

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    internal_messages: Annotated[list[AnyMessage], add_messages]
    tool_call: Optional[AIMessage] = None
    
    
# NODO AGENTE
def agent_node(state: State, config: dict) -> State:
    current_datetime = datetime.datetime(2024, 11, 8).strftime("%A, %d of %B, %Y")
    prompt = f"""
Eres un asistente que ayuda con la gestión de la bandeja de entrada del usuario Matias Barrera (matias.barrera@huemulsolutions.com)
La fecha de hoy es {current_datetime}
La conversación es la siguiente:
"""
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=prompt),
            MessagesPlaceholder("messages")
        ]
    )
    llm_w_tools = llm.bind_tools(config['configurable']['tools'])
    chain = chat_prompt | llm_w_tools
    response = chain.invoke({"messages": state['internal_messages']})
    tool_calls = response.additional_kwargs.get('tool_calls')
    if tool_calls:
        state['tool_call'] = tool_calls
    else:
        ai_message = AIMessage(content=response.content)
        state['messages'] = ai_message
        state['internal_messages'] = ai_message
    return state

# NODO DECISIÓN
def should_continue(state: State) -> str:
    if state.get('tool_call'):
        return 'tool'
    return 'message'
    
# NODO HERRAMIENTA
def tool_execution(state: State, config: dict) -> State:
    tool_calls = state['tool_call']
    function_name = tool_calls[0]['function']['name']
    args = json.loads(tool_calls[0]['function']['arguments'])
    
    tool = next((x for x in config['configurable']['tools'] if x.name == function_name), None)
    if tool:
        try:
            tool_output = tool.func(**args)
        except Exception as e:
            tool_output = f"Error al ejecutar la herramienta"
        tool_message = SystemMessage(content=f"Se ejecutó la herramienta {function_name} con los argumentos {args} y se obtuvo el siguiente resultado: {tool_output}")
        state['internal_messages'] = tool_message
    else:
        state['internal_messages'] = SystemMessage(content=f"No se encontró la herramienta {function_name}")
    state['tool_call'] = None
    return state


def create_graph() -> CompiledStateGraph:
    graph = StateGraph(State)

    graph.add_node('agent', agent_node)
    graph.add_node('tool', tool_execution)

    graph.set_entry_point('agent')
    graph.add_edge('tool', 'agent')
    graph.add_conditional_edges(
        'agent', should_continue, 
        {
            'tool': 'tool',
            'message': END
        }
    )
    compiled_graph = graph.compile()
    return compiled_graph

def execute_graph(graph: CompiledStateGraph, messages: list,
                  internal_messages: list, tools: list) -> State:
    state = {
        'messages': messages,
        'internal_messages': internal_messages,
        'tool_call': None
    }
    config = {
        'tools': tools
    }
    output = graph.invoke(state, config=config)
    return output
    
    
def conversation(user_input: str, messages: list, internal_messages: list) -> tuple:
    human_message = HumanMessage(content=user_input)
    messages.append(human_message)
    internal_messages.append(human_message)
    
    graph = create_graph()
    output = execute_graph(graph, messages, internal_messages, tools)
    print(output['messages'][-1].content)
    messages = output['messages']
    internal_messages = output['internal_messages']
    return messages, internal_messages


if __name__ == "__main__":
    user_input = ""
    messages = []
    internal_messages = []
    
    print("\nHola, ¿en qué puedo ayudarte? (/exit para salir)")
    while True:
        user_input = input("> ")
        if user_input == "/exit":
            print("Exiting...")
            break
        else:
            messages, internal_messages = conversation(user_input, messages, internal_messages)