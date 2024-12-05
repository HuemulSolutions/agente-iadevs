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
import uuid
import datetime, json
from enola.tracking import Tracking
from enola import evaluation
from enola.enola_types import EvalType, ErrOrWarnKind
from dotenv import load_dotenv

import os
load_dotenv()

NOMBRE_AGENTE = "Demo Agente UDD"
APP_ID = "enola-demo-udd"
USER_ID = "matias.barrera"
CHANNEL_ID = "console"

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
def agent_node(state: State, config:dict) -> State:
    print('NODO AGENTE')
    ########################################################
    # TRACKING ENOLA
    tracking: Tracking = config['configurable']['tracking']
    step = tracking.new_step(
        name="Agent Node",
        message_input=state['internal_messages'][-1].content
    )
    step.add_extra_info('Modelo', model)
    step.add_extra_info('Temperatura', temperature)
    ########################################################

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
    usage = response.usage_metadata
    
    ########################################################
    # TRACKING ENOLA
    step.add_extra_info('Prompt', chat_prompt.format(messages=state['internal_messages']))
    
    input_cost = usage['input_tokens'] * (0.005 / 1000)
    output_cost = usage['output_tokens'] * (0.015 / 1000)
    tracking.close_step_token(
        step=step, successfull=True,
        message_output=response.content if response.content else json.dumps(response.additional_kwargs),
        token_input_num=usage['input_tokens'],
        token_input_cost=input_cost,
        token_output_num=usage['output_tokens'],
        token_output_cost=output_cost,
        token_total_num=usage['total_tokens'],
        token_total_cost=input_cost + output_cost,
    )
    ########################################################
    
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
    print('NODO DECISIÓN')
    if state.get('tool_call'):
        return 'tool'
    return 'message'
    

# NODO HERRAMIENTA
def tool_execution(state: State, config: dict) -> State:
    print('NODO HERRAMIENTA')
    ########################################################
    # TRACKING ENOLA
    tracking: Tracking = config['configurable']['tracking']
    step = tracking.new_step(
        name="Tool Execution",
        message_input=json.dumps(state['tool_call'][0])
    )
    ########################################################
    
    tool_calls = state['tool_call']
    function_name = tool_calls[0]['function']['name']
    args = json.loads(tool_calls[0]['function']['arguments'])
    success = True
    
    tool = next((x for x in config['configurable']['tools'] if x.name == function_name), None)
    if tool:
        try:
            tool_output = tool.func(**args)
        except Exception as e:
            tool_output = f"Error al ejecutar la herramienta"
            success = False
            # Añadiendo error al tracking
            step.add_error(
                id=str(uuid.uuid4()),
                message=f"Error al ejecutar la herramienta {function_name} con los argumentos {args}, error: {e}",
                kind=ErrOrWarnKind.INTERNAL_CONTROLLED,
            )
        else:
            # Añadiendo resultado al tracking
            step.add_extra_info('Tool Output', json.dumps({"output": tool_output}))
        tool_message = SystemMessage(content=f"Se ejecutó la herramienta {function_name} con los argumentos {args} y se obtuvo el siguiente resultado: {tool_output}")
        state['internal_messages'] = tool_message
    else:
        success = False
        # Añadiendo error al tracking
        step.add_error(
            id=str(uuid.uuid4()),
            message=f"No se encontró la herramienta {function_name}",
            kind=ErrOrWarnKind.INTERNAL_CONTROLLED,
        )
        state['internal_messages'] = SystemMessage(content=f"No se encontró la herramienta {function_name}")
    state['tool_call'] = None
    
    # Cerrando step
    tracking.close_step_others(
        step=step, successfull=success,
    )
    return state


def create_graph()-> CompiledStateGraph:
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
    print('Compiling graph...')
    compiled_graph = graph.compile()
    return compiled_graph

def execute_graph(graph: CompiledStateGraph, messages: list,
                  internal_messages: list, tools: list, tracking: Tracking)-> tuple:
    state = {
        'messages': messages,
        'internal_messages': internal_messages,
        'tool_call': None
    }
    
    config = {
        'tools': tools,
        'tracking': tracking,
    }
    output = graph.invoke(state, config=config)
    return output, tracking


def conversation_(user_input: str, messages: list, internal_messages: list, session_id: str)-> tuple:
    # INICIANDO TRACKING ENOLA
    tracking = Tracking(
                token=os.getenv("ENOLA_TOKEN_AGENTE"),
                name=NOMBRE_AGENTE,
                app_id=APP_ID,
                user_id=USER_ID,
                channel_id=CHANNEL_ID,
                is_test=True,
                session_id=session_id,
                message_input=user_input
            )
    human_message = HumanMessage(content=user_input)
    messages.append(human_message)
    internal_messages.append(human_message)
    
    graph = create_graph()
    output, tracking = execute_graph(graph, messages, internal_messages, tools, tracking)
    
    # CERRANDO TRACKING
    tracking.execute(
        successfull=True,
        message_output=output['messages'][-1].content,
    )
    
    print(output['messages'][-1].content)
    messages = output['messages']
    internal_messages = output['internal_messages']
    return messages, internal_messages, tracking.enola_id


def user_evaluation(enola_id: str)-> None:
    user_eval = input("Del 1 al 5, ¿qué tan útil fue la respuesta?: ")
    try:
        user_eval = int(user_eval)
    except ValueError:
        print("El valor ingresado no es válido")
        return
    if user_eval < 1 or user_eval > 5:
        print("El valor ingresado no es válido")
        return
    
    user_comment = input("¿Algún comentario adicional?: ")
    eval = evaluation.Evaluation(
        token=os.getenv("ENOLA_TOKEN_AGENTE"),
        eval_type=EvalType.USER,
        user_id=USER_ID,
    )
    eval.add_evaluation_by_level(
        enola_id=enola_id, # Se obtiene el enola_id de la evaluación
        eval_id="0",
        level=user_eval,
        comment=user_comment,
    )
    _ = eval.execute()



if __name__ == "__main__":
    user_input = ""
    messages = []
    internal_messages = []
    enola_id = None
    session_id = str(uuid.uuid4())
    
    print("\nHola, ¿en qué puedo ayudarte? (/exit para salir, /eval para evaluar después de una respuesta)")
    while True:
        user_input = input("> ")
        if user_input == "/exit":
            print("Exiting...")
            break
        elif user_input == "/eval":
            if not enola_id:
                print("Primero debes iniciar una conversación para evaluar")
            else:
                user_evaluation(enola_id)
                print("Gracias por tu evaluación, puedes continuar con la conversación!")
        else:
            messages, internal_messages, enola_id = conversation_(user_input, messages, internal_messages, session_id)
            
            
            
            
    