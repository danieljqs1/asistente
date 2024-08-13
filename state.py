import operator
from typing import Annotated, TypedDict, Union
from langgraph.graph.message import add_messages
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: Annotated[list, add_messages]
