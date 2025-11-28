from typing import Annotated, Literal, TypedDict

from langgraph.graph import MessagesState


class Action(TypedDict):
    action: str


class Step(TypedDict):
    step_number: int
    reasoning: list[str]
    actions: list[Action]


def step_reducer(old_steps: list[Step], new_steps: list[Step]):
    merged_steps = list[Step]()
    mapping = dict[int, tuple[list[str], list[Action]]]()
    for old_step in old_steps:
        value = mapping.setdefault(old_step["step_number"], (list[str](), list[Action]()))
        value[1].extend(old_step["actions"])
        value[0].extend(old_step["reasoning"])
    for new_step in new_steps:
        value = mapping.setdefault(new_step["step_number"], (list[str](), list[Action]()))
        value[1].extend(new_step["actions"])
        value[0].extend(new_step["reasoning"])

    for step_number in sorted(list(mapping.keys())):
        merged_steps.append(Step(step_number=step_number, reasoning=mapping[step_number][0], actions=mapping[step_number][1]))

    return merged_steps


class BaseAgentState(MessagesState):
    current_step: int
    steps: Annotated[list[Step], step_reducer]
    question: str
    answer: str | None


class CodeExecutionAction(Action):
    action: Literal["code_execution"]
    code: str
    output: str