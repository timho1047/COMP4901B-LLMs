import os
from typing import List, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from prompts import (
    CODE_EXECUTION_AGENT_SYSTEM_PROMPT,
    NO_CODE_EXECUTION_AGENT_SYSTEM_PROMPT,
)
from schema import BaseAgentState, Step
from tools import execute_python_code, submit_answer

load_dotenv()


def create_agent[S: BaseAgentState](state_cls: type[S], system_prompt: str, tools: List[BaseTool]):
    llm = init_chat_model(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0.6,
        max_tokens=8192,
    ).bind_tools(tools)

    def invoke_llm(messages: List[BaseMessage], state: S, max_retries: int = 3) -> AIMessage:
        for _ in range(max_retries):
            try:
                ai_message = llm.invoke(messages)
                if ai_message.invalid_tool_calls:
                    raise Exception(f"Invalid tool calls: {ai_message.invalid_tool_calls}")
                return ai_message
            except Exception as e:
                print(f"Error in answering question {state['question']}")
                print(f"Error invoking LLM, retrying...: {e}")
                continue
        raise Exception(f"Failed to invoke LLM after {max_retries} retries")

    def agent(state: S, config: RunnableConfig) -> dict:
        if state["current_step"] >= config["configurable"]["max_steps"]:
            return {
                "messages": [
                    HumanMessage(
                        content=f"<system_reminder>Reached the maximum number of steps ({config["configurable"]["max_steps"]}). Stop.</system_reminder>"
                    )
                ],
                "answer": "\\boxed{failure}",
            }

        if len(state["messages"]) == 0:  # At the beginning of the conversation
            human_message = HumanMessage(content=state["question"])
            ai_message = invoke_llm(
                [
                    SystemMessage(content=system_prompt),
                    human_message,
                ],
                state=state,
            )
            return {
                "messages": [human_message, ai_message],
                "current_step": state["current_step"] + 1,
                "steps": [
                    Step(
                        step_number=state["current_step"] + 1,
                        reasoning=[ai_message.content],
                        actions=[],
                    )
                ],
            }
        else:
            system_reminder = f"<system_reminder>The current question your are investigating is: [{state['question']}]. If you have not yet found out the answer, please ignore this system reminder and continue to make use of tools provided to you (if any) to gather more information and think about how to answer the question. If you are confident that you have found out the answer, please use `submit_answer` tool to submit the answer concisely.</system_reminder>"

            ai_message = invoke_llm(
                [
                    SystemMessage(content=system_prompt),
                    *state["messages"],
                    HumanMessage(content=system_reminder),
                ],
                state=state,
            )
            return {
                "messages": [ai_message],
                "current_step": state["current_step"] + 1,
                "steps": [
                    Step(
                        step_number=state["current_step"] + 1,
                        reasoning=[ai_message.content],
                        actions=[],
                    )
                ],
            }

    def should_continue(state: S) -> Literal["agent", END]:
        if state["answer"] is not None:
            return END
        return "agent"

    return (
        StateGraph(state_cls)
        .add_node("agent", agent)
        .add_node("tools", ToolNode(tools))
        .add_edge(START, "agent")
        .add_edge("agent", "tools")
        .add_conditional_edges("tools", should_continue)
        .compile(checkpointer=InMemorySaver())
    )


def create_code_execution_agent():
    return create_agent(
        BaseAgentState,
        CODE_EXECUTION_AGENT_SYSTEM_PROMPT,
        [execute_python_code, submit_answer],
    )


def create_no_code_execution_agent():
    return create_agent(BaseAgentState, NO_CODE_EXECUTION_AGENT_SYSTEM_PROMPT, [submit_answer])
