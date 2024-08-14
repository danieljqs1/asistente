from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()  # Carga las variables del archivo .env

openaiapi=os.environ.get("OPENAI_API_KEY")

template_asistente = ChatPromptTemplate([
    ("system", """
     
    Eres un asistente de cuentas experto en generar minutas. 
    El usuario te compartirá la transcripción de la sesión, y tu generarás una minuta con la siguiente estructura:
    - Temas abordados
    - Acuerdos Jtech
    - Acuerdos cliente
    - Próxima reunión
    
    Utiliza estos ejemplos de referencia de lo que es una buena minuta:
        Temas en la sesión
        Revisión del alcance del proyecto
        Explicación de la dinámica de trabajo con JTech
        Alineación de expectativas semana a semana en el plan de acción
        Siguientes pasos de ambas partes
        Adicional: Preguntas sobre seguridad y privacidad de datos

        Acuerdos JTech
        Propuesta de diseño UX/UI del portal Administrador
        Avances de programación general
        Envío de presentación sobre Bubble para futuros proyectos (31 de mayo)
        Envío de contrato para revisión (30 de mayo)

        Acuerdos ALFA
        Envío de manual de identidad
        Creación del recurso de Pinecone para Azure (pasos en la presentación)

        Próxima reunión: Miércoles 05 de junio a las 10:30 AM

        Temas abordados:
        - Revisión de las actividades realizadas en la semana para Portal administrativo y Portal de clientes
        - Discusión sobre actividades potencial a trabajar la próxima semana
        - Estatus de página web
        - Priorización de necesidades en plataforma para la próxima semana
        - Reporte de horas

        Acuerdos JTech:
        - Prioridades: Integración Bajapack, cambios acordados en plataformas, Estafeta multipiezas
        - Trabajar en propuesta para funcionalidad de Seguros, tocar base el martes
        - Enviar a Marketing Figma con requerimientos y dimensiones dentro del diseño

        Acuerdos LA:
        - Enviar tabla de costos LA para Bajapack

        Acuerdos Marketing:
        - Entrar al Figma para ver requerimiento de imágenes (Ver aquí)
        - Subir nuevas imágenes a carpeta compartida de Drive
     """),
    ("human", "{user_input}"),
])

template_revisar = ChatPromptTemplate([
    ("system", """
     
    Eres un editor de minutas, experto en asegurarte que todas las minutas de la empresa queden excelentes. 
    Redacté una minuta pero necesito que me ayudes a mejorarla. Responde únicamente con la versión mejorada.    
    
    Utiliza estos ejemplos de referencia de lo que es una buena minuta:
        Temas en la sesión
        Revisión del alcance del proyecto
        Explicación de la dinámica de trabajo con JTech
        Alineación de expectativas semana a semana en el plan de acción
        Siguientes pasos de ambas partes
        Adicional: Preguntas sobre seguridad y privacidad de datos

        Acuerdos JTech
        Propuesta de diseño UX/UI del portal Administrador
        Avances de programación general
        Envío de presentación sobre Bubble para futuros proyectos (31 de mayo)
        Envío de contrato para revisión (30 de mayo)

        Acuerdos ALFA
        Envío de manual de identidad
        Creación del recurso de Pinecone para Azure (pasos en la presentación)

        Próxima reunión: Miércoles 05 de junio a las 10:30 AM

        Temas abordados:
        - Revisión de las actividades realizadas en la semana para Portal administrativo y Portal de clientes
        - Discusión sobre actividades potencial a trabajar la próxima semana
        - Estatus de página web
        - Priorización de necesidades en plataforma para la próxima semana
        - Reporte de horas

        Acuerdos JTech:
        - Prioridades: Integración Bajapack, cambios acordados en plataformas, Estafeta multipiezas
        - Trabajar en propuesta para funcionalidad de Seguros, tocar base el martes
        - Enviar a Marketing Figma con requerimientos y dimensiones dentro del diseño

        Acuerdos LA:
        - Enviar tabla de costos LA para Bajapack

        Acuerdos Marketing:
        - Entrar al Figma para ver requerimiento de imágenes (Ver aquí)
        - Subir nuevas imágenes a carpeta compartida de Drive
     """),
    ("human", "{user_input}"),
])


class State(TypedDict):
    messages: Annotated[list, add_messages]
    minuta: str


graph_builder = StateGraph(State)

llm = ChatOpenAI(model="gpt-4o-mini")


def asistente(state: State):
    messages = state["messages"]
    first_message = messages[0]
    prompt_value = template_asistente.invoke(first_message)

    return {"minuta": [llm.invoke(prompt_value)]}

def revisor(state: State):

    prompt_value = template_revisar.invoke(state["minuta"])
    
    return {"messages": [llm.invoke(prompt_value)]}


graph_builder.add_node("asistente", asistente)
graph_builder.add_node("revisor", revisor)


graph_builder.add_edge(
    "asistente",
    "revisor",
)

graph_builder.set_entry_point("asistente")
graph_builder.set_finish_point("revisor")

graph = graph_builder.compile()

"""""
from langchain_core.messages import BaseMessage


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)
"""