from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from app.llm import llm
from app.prompts import SYSTEM_PROMPT

conversation_history = [
    SystemMessage(content=SYSTEM_PROMPT)
]


def chat_service(message: str):

    conversation_history.append(
        HumanMessage(content=message)
    )

    response = llm.invoke(conversation_history)

    conversation_history.append(
        AIMessage(content=response.content)
    )

    return response.content


def clear_chat_service():

    global conversation_history

    conversation_history = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]