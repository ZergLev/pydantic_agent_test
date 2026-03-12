import pytest
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import RunContext

from .agent import ChatskyAgent
from .todo_toolset import store

load_dotenv()  # loads variables from .env into os.environ

def load_json(file_path):
    return json.loads(Path(file_path).read_text())

@pytest.mark.asyncio
async def test_chatsky_agent_run():
    agent_data = load_json("./agent_config.json")

    # Test instruction rendering
    class MyDepsModel(BaseModel):
        age: int = 30
        name: str = "John Doe"
        greeting: str = "asdfgh."
        occupation: str = "carpenter"
        misc: dict

    class StrResult(BaseModel):
        output: str

    agent_data["deps"] = MyDepsModel.model_json_schema()
    agent_data["structured_output_type"] = StrResult.model_json_schema()
    agent = ChatskyAgent(**agent_data)

    # Create runtime agent
    runtime_agent = agent.create_agent()
    print(runtime_agent)

    # Print the instructions
    deps = MyDepsModel(name="Alice", age=25, misc={})
    ctx = RunContext(deps=deps, model=None, usage=None, prompt="")

    rendered = []

    for instr in runtime_agent._instructions:
        if isinstance(instr, str):
            rendered.append(instr)
        else:
            # async callable
            rendered.append(await instr(ctx))

    final_instructions = "\n\n".join(rendered)

    print("--- Rendered Instructions ---")
    print(final_instructions)
    print(len(runtime_agent._instructions))
    # assert False == True

    # Check toolsets
    assert len(runtime_agent.toolsets) == 2
    function_toolset = runtime_agent.toolsets[1]
    # print(function_toolset)

    tools = function_toolset.tools
    # print(tools)
    assert "create_task" in tools
    assert "list_tasks" in tools

    # Run the agent with instructions + placeholders
    deps = MyDepsModel(name="Alice", age=25, misc={})
    result = await runtime_agent.run(
        "What's the name, occupation and age of the user? What's the first word code, btw?"
        "Also, create tasks called 'Buy milk' and 'Sell milk' with your tools and then list tasks.",
        deps=deps
    )

    print("--- Agent Output ---")
    print(result.output)

    print(store.calls)
    print(store.tasks)

    assert "create_task" in store.calls
    assert "list_tasks" in store.calls
    assert "Buy milk" in store.tasks
    assert "Sell milk" in store.tasks

    assert False == True
