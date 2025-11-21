# COMP4901B Homework 4 - AIME Math Problem Solver with Python Tool

**Due Date: 2025-12-5 11:59 PM**

**Full score: 100 points.**

This is a small homework that extends your search agent in the group project with Python code execution capabilities to solve challenging AIME problems. You can view this as a small addon to your group project, thus your codebase for the group project can be largely reused for this homework to make things easier (or vice versa). 

## Project Overview

In this homework, you will **build on top of your search agent implementation** from the group project (you can start this homework first and build the group project based on it, which is fine as well) and add a **Python code execution tool**. Your agent will:
1. **Reason** about complex AIME math problems
2. **Execute Python code** to perform calculations, verify solutions, and explore patterns
3. **Iterate** through multiple reasoning steps

The key addition is the **`execute_python_code` tool** that allows your agent to:
- Perform symbolic and numerical computations (using sympy, numpy, etc.)
- Verify mathematical solutions
- Explore patterns and test hypotheses
- Break down complex problems into computational steps


## Project Structure

```
assignment4/
├── README.md                      # This file
├── pyproject.toml                # Python dependencies
├── src/                           # YOUR AIME agent with Python tool
├── scripts/
│   └── evaluate_aime.py          # AIME evaluation script (PROVIDED)
├── data/
│   └── aime/                      # AIME dataset 
```

## Getting Started

### 1. Prerequisites

**Important**: This homework assumes you are already familar with the group project -- if not, you can check the group project readme for more detailed description of your setup. 

### 2. Installation

> Similar to the group project, you are allowed to use other package managers like conda, pip, etc., but below we only project instructions of uv for an example 

**Step 1: Install uv (if not already installed)**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Step 2: Create Environment and Install Dependencies**

```bash
# Create a virtual environment with uv
uv sync

# Activate the environment
source .venv/bin/activate 
```

**Note:** After activating the environment with source .venv/bin/activate, you can run all Python commands normally (e.g., python script.py). The environment stays active in your current terminal session.

**Step 3: Set Up API Key**

For this project, you'll use the latest DeepSeek-v3.2 model for LLMs. We have provided these keys in Canvas announcements for the group project. 

**Important: Please keep these keys confidential and do not share them with anyone outside the class. Also, if use public repos to store your code, please make sure to remove the keys from the code before pushing to the repo.**


## Your Tasks

### Step 1: Implement the Python Code Execution Tool (25 Points)

Building on your search agent from the group project, you need to add a **Python code execution tool**. This tool should:

1. **Accept Python code** as a string parameter
2. **Execute the code** in a safe environment
3. **Return the output** (stdout, stderr, and return value)
4. **Handle errors** gracefully

**Tool Definition Example:**
```python
{
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": "Execute Python code to perform calculations and verify solutions",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
}
```

**Available Python modules** for your code execution environment (at least):
- `math`: Standard mathematical functions
- `fractions`: Rational number arithmetic
- `itertools`: Iterator building blocks
- `sympy`: Symbolic mathematics
- `numpy`: Numerical computing

#### Report in the PDF
* Brief explanation of your implementation of this python tool. 
* Screanshot the core implementation code.  

### Step 2: Adapt Your Agent Loop for AIME (25 Points) 

Similar to your group project (**We highly recommend you using the same codebase to the group project, because they should be very similar**), implement an agent that

1. **Initialize** with a system prompt for mathematical problem-solving
2. **Loop** for up to `max_steps` (recommended: 20 for multi-step reasoning)
3. **Support tool calling** for Python code execution
4. **Track reasoning steps** and conversation history



Your system prompt should:
- Explain that the agent is solving AIME problems
- Encourage use of Python for calculations and verification
- Instruct to format final answers in `\boxed{}` format
- Promote step-by-step reasoning
- List available Python modules

**Example prompt :**

1. Without tool calling:
```python
messages = [
                {
                    "role": "system",
                    "content": """You are solving AIME (American Invitational Mathematics Examination) problems. Put your final answer in \\boxed{} format."""
                },
                {
                    "role": "user",
                    "content": problem
                }
            ]
```
2. With tool calling: 

```python
messages = [
            {
                "role": "system",
                "content": """You are solving AIME (American Invitational Mathematics Examination) problems.

You have access to Python code execution via the execute_python_code tool. Use it to perform calculations, verify solutions, or explore patterns.

Put your final answer in \\boxed{} format.

Available Python modules: math, fractions, itertools, sympy, numpy"""
            },
            {
                "role": "user",
                "content": problem
            }
        ]
```
#### Report in the PDF
* Brief explanation of your implementation of this agent. 

### Step 3: Evaluate your agent w/ and w/o tool calling (50 Points) 

You must evaluate your agent in **TWO settings**:

#### Setting 1: Without Tool (Baseline) (25 Points) 
- **Configuration**: No Python tool available, direct answer generation
- **Max turns**: 1 (single-turn generation)
- **Num rollout**: at least 4 (The number of rollout for each question)
- **Temperature**: 0.6 
- **Model**: deepseek-chat
- **Expected accuracy**: ≥ 0.60 (60%)

This establishes a baseline of the LLM's mathematical reasoning without computational tools.

#### Setting 2: With Python Tool (25 Points) 
- **Configuration**: Python code execution tool enabled
- **Max turns**: 20 (multi-step reasoning with tool calls)
- **Num rollout**: at least 4 (The number of rollout for each question)
- **Temperature**: 0.6
- **Model**: deepseek-chat
- **Expected accuracy**: ≥ 0.70 (70%)

This demonstrates the improvement from tool-augmented reasoning.

#### Evaluation Script

Use the provided evaluation script to assess your results:

```bash
python scripts/evaluate_aime.py \
    --input results/your_predictions.jsonl \
    --output results/evaluation.jsonl
```


#### Required Prediction Format

Your predictions file must be in JSONL format with the following fields for each problem:

```jsonl
{"id": 60, "rollout_id": 0, "problem": "Every morning Aya goes for a...", "answer": "204", "llm_response": "The answer is \\boxed{204}", ...}
```

**Required fields:**
- `id`: Problem ID
- `rollout_id`: Rollout number (0 for first attempt)
- `problem`: The problem text
- `answer`: Gold standard answer
- `llm_response`: Your model's final response output


#### Report
* Screenshot or plot of accuracy of these two settings. 


## Submission Guidelines

Submit a PDF report and a zip of the codebase (only the homework directory) to Canvas. Your PDF should include your full name, student ID.

**Important:** Please submit the codebase zip and PDF report in two seperate files.


