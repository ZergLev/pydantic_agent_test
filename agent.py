import logging
import json
import json_schema_to_pydantic
from pathlib import Path
from typing import Any, Type
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_ai import Agent, RunContext, AbstractToolset

from .models import ChatskyToolset, AdditionalConfiguration
from .todo_toolset import TodoToolset

available_toolsets: dict[str, Any] = {
    "todo_toolset": TodoToolset
}

logger = logging.getLogger(__name__)

class ChatskyAgent(BaseModel, arbitrary_types_allowed=True):
    name: str
    description: str | None = None
    instructions: list[str] = Field(default_factory=list)  # local instructions
    toolsets: list[ChatskyToolset] = Field(default_factory=list)
    structured_output_type: dict | None = None  # JSON schema
    deps: dict  # JSON schema
    model: str
    additional_configuration: AdditionalConfiguration


    def create_deps_model(self) -> Type[BaseModel]:
        return json_schema_to_pydantic.create_model(self.deps)

    def create_result_model(self) -> Type[BaseModel] | None:
        if not self.structured_output_type:
            return None
        return json_schema_to_pydantic.create_model(self.structured_output_type)

    def create_toolsets(self) -> list[AbstractToolset]:
        instances = []
        for ts in self.toolsets:
            if ts.name not in available_toolsets:
                raise ValueError(f"Unknown toolset: {ts.name}")

            toolset_class = available_toolsets[ts.name]
            toolset = toolset_class(
                config=ts.configuration,
                approval_required=ts.approval_required
            )

            instances.append(toolset)
        return instances


    def create_agent(self) -> Agent:
        deps_model = self.create_deps_model()
        result_model = self.create_result_model()
        toolsets = self.create_toolsets()

        agent = Agent(
            name=self.name,
            model=self.model,
            deps_type=deps_model,
            output_type=result_model,
            toolsets=toolsets or None,
            model_settings=self.additional_configuration.model_settings,
            # usage_limits=self.additional_configuration.usage_limits,
            max_concurrency=self.additional_configuration.concurrency_limit,
        )
        @agent.instructions
        async def render_instruction(ctx: RunContext[deps_model]) -> str:

            deps_values = ctx.deps.model_dump()

            # Skip undefined placeholders with no defaults
            context = {k: v for k, v in deps_values.items() if v is not None}

            rendered = []
            for instr in self.instructions:
                try:
                    rendered.append(instr.format(**context))
                except KeyError as e:
                    logger.debug(
                        f"Skipping instruction '{instr}' - placeholder not found: {e}"
                    )
                    continue

            return "\n\n".join(rendered)

        return agent

