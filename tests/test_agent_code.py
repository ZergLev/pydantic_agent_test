import pytest
import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import RunContext

from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart

from ..agent import ChatskyAgent
from ..todo_toolset import TodoToolset

load_dotenv()

def load_json(file_path):
    return json.loads(Path(file_path).read_text())

@pytest.mark.asyncio
async def test_create_deps_model_1():
    agent_data = load_json("./tests/configs/test_create_deps.json")
    agent = ChatskyAgent(**agent_data)

    deps_model = agent.create_deps_model()
    assert issubclass(deps_model, BaseModel)

    instance = deps_model(age=25)
    assert instance.name == "John"
    assert instance.age == 25

@pytest.mark.asyncio
async def test_create_deps_model_2():

    class MyDepsModel(BaseModel):
        age: int = 30
        name: str = "John"

    agent_data = load_json("./tests/configs/test_create_deps.json")
    agent_data["deps"] = MyDepsModel.model_json_schema()

    agent = ChatskyAgent(**agent_data)

    deps_model = agent.create_deps_model()
    assert deps_model.model_json_schema() == MyDepsModel.model_json_schema()

    instance = deps_model(age=25)
    assert instance.name == "John"
    assert instance.age == 25

# Made for `TodoToolset`
@pytest.mark.asyncio
async def test_create_toolsets():
    agent_data = load_json("./tests/configs/test_create_toolsets.json")
    agent = ChatskyAgent(**agent_data)

    toolsets = agent.create_toolsets()
    assert len(toolsets) == 1

    toolset = toolsets[0]
    assert isinstance(toolset, TodoToolset)
    assert toolset.max_tasks == 5
    assert toolset.approval_required == ["create_task"]

@pytest.mark.asyncio
async def test_create_unknown_toolset():
    agent_data = load_json("./tests/configs/test_create_toolsets.json")

    fake_toolset = {"name": "nonexistent_toolset"}
    agent_data["toolsets"].append(fake_toolset)

    agent = ChatskyAgent(**agent_data)

    with pytest.raises(ValueError):
        agent.create_toolsets()

@pytest.mark.asyncio
async def test_create_result_model_valid():
    agent_data = load_json("./tests/configs/test_create_result_model.json")
    agent = ChatskyAgent(**agent_data)

    result_model = agent.create_result_model()
    assert issubclass(result_model, BaseModel)

    instance = result_model()
    assert instance.output == "hello"
    assert instance.score == 0

@pytest.mark.asyncio
async def test_create_result_model_none():
    agent_data = load_json("./tests/configs/test_create_result_model.json")
    agent_data["structured_output_type"] = None
    agent = ChatskyAgent(**agent_data)

    result_model = agent.create_result_model()

    assert result_model is None

@pytest.mark.asyncio
async def test_instruction_rendering():

    def echo_instructions(messages, info: AgentInfo):
        return ModelResponse(
            parts=[
                TextPart(json.dumps({
                    "response": info.instructions
                }))
            ]
        )
    echo_model = FunctionModel(echo_instructions)

    agent_data = load_json("./tests/configs/test_instructions.json")
    agent = ChatskyAgent(**agent_data)
    runtime_agent = agent.create_agent()

    # print(runtime_agent.output_type)
    # print(runtime_agent.toolsets)

    deps_model = agent.create_deps_model()
    deps = deps_model(name="Alice", age=25)

    with runtime_agent.override(model=echo_model):
        result = await runtime_agent.run(
            "Hello! What's the user's name?",
            deps=deps
        )

    output = result.output.response.strip()

    assert "job" not in output
    assert "country" not in output

    expected = "Hello Alice\n\nYou are 25 years old"
    assert output == expected

# TODO: consider either removing this test or mocking agent behaviour
@pytest.mark.asyncio
async def test_full_agent_run():
    # assert False
    agent_data = load_json("./tests/configs/test_full.json")

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
    print(agent_data["deps"])

    # Create runtime agent
    runtime_agent = agent.create_agent()
    print(runtime_agent)

    # Print the instructions
    deps = MyDepsModel(name="Alice", age=25, misc={})
    ctx = RunContext(deps=deps, model=None, usage=None, prompt="")

    rendered = []

    for instr in runtime_agent._instructions:
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
        "Also, create tasks called 'Buy milk', another 'Buy milk' and 'Sell milk' with your tools and then list tasks.",
        deps=deps
    )

    print("--- Agent Output ---")
    print(result.output)

    # assert isinstance(result.output, str)
    assert len(str(result.output)) > 0

    # assert False == True
