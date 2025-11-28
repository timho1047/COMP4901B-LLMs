import re
import subprocess

from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from schema import CodeExecutionAction, Step


@tool
def submit_answer(content: str, runtime: ToolRuntime):
    """Provide the final short answer.

    Args:
        content: The short answer text. It must end with the final answer wrapped in a latex box, e.g. "The velocity is $\boxed{42}$".
    """

    # Check if the mandatory \boxed{ format is present
    if "\\boxed{" not in content:
        return (
            "Error: The answer is not formatted with \\boxed{}. "
            'Please wrap your final numeric answer in \\boxed{}, e.g., "The answer is \\boxed{42}".'
        )

    # Basic check to ensure the box isn't empty
    match = re.search(r"\\boxed\{(.*?)\}", content, re.DOTALL)
    if match and not match.group(1).strip():
        return "Error: The \\boxed{} tag is empty. Please put the answer inside."

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content="The final solution is successfully submitted: " + content,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "answer": content,
        }
    )


def validate_code(code: str):
    """
    Validates the Python code by checking for forbidden keywords.
    """

    IMPORT_PATTERN = r"^import (\w+)\.?"
    FROM_PATTERN = r"^from (\w+)\.?"
    ALLOWED_MODULES = ["math", "fractions", "itertools", "sympy", "numpy"]
    FORBIDDEN_CALLS = ["open(", "input(", "eval(", "exec(", "compile(", "__import__"]

    if "print(" not in code:
        raise ValueError(
            "Print statement `print()` is required. Otherwise, the code execution will not return any output."
        )

    for line in code.split("\n"):
        if m := re.match(IMPORT_PATTERN, line) or re.match(FROM_PATTERN, line):
            if m[1] not in ALLOWED_MODULES:
                raise ValueError(
                    f"You are not allowed to import {m[1]}. Use the provided modules: math, fractions, itertools, sympy (alias sp), numpy (alias np)."
                )

    for call in FORBIDDEN_CALLS:
        if call in code:
            raise ValueError(f"keyword '{call}' is not allowed.")


CODE_TEMPLATE = """
import math
import fractions
import itertools
import sympy
import sympy as sp
import numpy
import numpy as np

{code}
"""


def execute_python_code_impl(code: str):
    try:
        validate_code(code)

        code_block = CODE_TEMPLATE.format(code=code)
        result = subprocess.run(
            ["uv", "run", "python", "-c", code_block],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return result.stdout
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (limit: 30 seconds)."
    except subprocess.CalledProcessError as e:
        return f"Error executing code:\n{e.stderr}"
    except ValueError as e:
        return f"Security Error: {str(e)}"
    except Exception as e:
        return f"Error executing Python code: {e}"


@tool
def execute_python_code(code: str, runtime: ToolRuntime):
    """
    Execute Python code to perform calculations, verify solutions, and explore patterns.

    This tool allows you to run Python scripts in a sandboxed environment.
    Use it to:
    - Perform complex calculations (symbolic or numerical)
    - Verify mathematical derivations
    - Brute-force small search spaces using itertools
    - Solve equations using sympy

    The following modules are PRE-IMPORTED and available for use:
    - `math`: Standard mathematical functions
    - `fractions`: Rational number arithmetic
    - `itertools`: efficient looping
    - `sympy` (aliased as `sp`): Symbolic mathematics (simplify, solve, etc.)
    - `numpy` (aliased as `np`): Numerical arrays and operations
    ```

    **Usage Rules:**
    1. **NO Imports**: Do not other modules than the pre-imported modules.
    2. **Print Results**: The tool captures `stdout`. You MUST `print()` variables to see them.
    3. **No I/O**: `input()`, `open()`, and file system access are forbidden.
    4. **Stateless**: Each execution is independent. Variables do not persist between calls. Include all necessary logic/variables in `code`.

    Args:
        code: The Python code to execute.
    """
    result = execute_python_code_impl(code)
    content = result.strip()

    if not content:
        content = "<system_reminder>The code executed successfully but returned no output. Did you print() the result correctly?</system_reminder>"

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=content,
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "steps": [
                Step(
                    step_number=runtime.state["current_step"],
                    reasoning=[],
                    actions=[
                        CodeExecutionAction(action="code_execution", code=code, output=result)
                    ],
                )
            ],
        }
    )


if __name__ == "__main__":
    code = """
import sympy as sp

# Given values
s = 5/2  # km/h
t = 24   # minutes

# New speed
new_speed = s + 1/2
print(f"New speed: {new_speed} km/h")

# Walking time at new speed
walking_time_hours = 9 / new_speed
walking_time_minutes = walking_time_hours * 60
print(f"Walking time: {walking_time_minutes} minutes")

# Total time including coffee shop
total_time = walking_time_minutes + t
print(f"Total time: {total_time} minutes")

# Verify it's an integer
print(f"Is total time an integer? {total_time.is_integer()}")
"""
    print(execute_python_code_impl(code))
