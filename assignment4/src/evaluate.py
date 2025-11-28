import argparse
import itertools
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from pprint import pprint
from typing import Literal

from tqdm import tqdm

from agent import create_code_execution_agent, create_no_code_execution_agent
from schema import BaseAgentState

RECURSION_LIMIT = 50
MAX_STEPS = 20
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "aime24.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "results"
ROLLOUTS = 4


def evaluate_single_question(
    id: int,
    question: str,
    ground_truth: str,
    rollout_id: int = 0,
    agent_type: Literal["nocode", "code"] = "nocode",
    enable_streaming: bool = False,
):
    if agent_type == "nocode":
        agent = create_no_code_execution_agent()
    elif agent_type == "code":
        agent = create_code_execution_agent()
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    init_state = BaseAgentState(
        messages=[],
        current_step=0,
        question=question,
        answer=None,
        steps=[],
    )

    config = {
        "configurable": {
            "thread_id": str(id),
            "max_steps": MAX_STEPS if agent_type == "code" else 1,
        },
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        if enable_streaming:
            for chunk in agent.stream(init_state, config=config, stream_mode="updates"):
                if "agent" in chunk:
                    if "current_step" in chunk["agent"]:
                        print("=======================")
                        print(f"=Current step: {chunk['agent']['current_step']}")
                        print("=======================")
                    for msg in chunk["agent"]["messages"]:
                        msg.pretty_print()

                if "tools" in chunk:
                    for msg in chunk["tools"]["messages"]:
                        msg.pretty_print()
            state: BaseAgentState = agent.get_state(config=config).values
        else:
            state: BaseAgentState = agent.invoke(init_state, config=config)
    except Exception as e:
        state = agent.get_state(config=config).values
        print("================================================")
        pprint(state)
        print("================================================")
        print(f"Error: \n{e}\n")
        raise e

    trajectory = {
        "id": id,
        "rollout_id": rollout_id,
        "question": question,
        "ground_truth": ground_truth,
        "trajectory": {
            "question": question,
            "steps": state["steps"],
            "final_answer": state["answer"],
            "total_search_steps": len(state["steps"]),
        },
    }

    prediction = {
        "id": id,
        "rollout_id": rollout_id,
        "question": question,
        "answer": ground_truth,
        "llm_response": state["answer"],
    }

    return trajectory, prediction


def evaluate_batch_questions(
    questions: list[dict[Literal["id", "question", "answer"], str | int]],
    agent_type: Literal["search", "raw"] = "search",
    n_rollouts: int = 4,
):
    with ThreadPoolExecutor(max_workers=60) as executor:
        futures = [
            executor.submit(
                evaluate_single_question,
                question["id"],
                question["question"],
                question["answer"],
                rollout_id,
                agent_type,
            )
            for question, rollout_id in itertools.product(questions, range(n_rollouts))
        ]
        results = [future.result() for future in tqdm(futures, desc="Evaluating questions")]
    return results


def load_questions(
    input_file: str,
) -> list[dict[Literal["id", "question", "answers"], str | int]]:
    with open(input_file, "r") as f:
        return [json.loads(line) for line in f.readlines()]


def save_results(results: list[tuple[dict, dict]], output_dir: str, run_name: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_file = output_dir / f"trajectories_{run_name}.jsonl"
    prediction_file = output_dir / f"prediction_{run_name}.jsonl"

    with trajectory_file.open("w") as t, prediction_file.open("w") as p:
        for trajectory, prediction in results:
            t.write(json.dumps(trajectory) + "\n")
            p.write(json.dumps(prediction) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--agent_type", type=str, required=True)
    args = parser.parse_args()

    questions = load_questions(INPUT_FILE)
    results = evaluate_batch_questions(questions, args.agent_type, n_rollouts=ROLLOUTS)
    save_results(results, OUTPUT_DIR / args.run_name, args.run_name)

    subprocess.run(
        [
            "uv",
            "run",
            "scripts/evaluate_aime.py",
            "--input",
            OUTPUT_DIR / args.run_name / f"prediction_{args.run_name}.jsonl",
            "--output",
            OUTPUT_DIR / args.run_name / f"evaluation_{args.run_name}.jsonl",
        ]
    )
