import logging
import json
import json_schema_to_pydantic
from pathlib import Path
from typing import Any, Type
from pydantic import BaseModel, Field, PrivateAttr
from pydantic_ai import Agent, RunContext, AbstractToolset

from .models import ChatskyToolset, AdditionalConfiguration
from .todo_toolset import some_toolset

# TODO: Add the toolset to tests
available_toolsets: dict[str, Any] = {
    "todo_toolset": some_toolset
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

    _global_instructions: list[str] = PrivateAttr(default_factory=list)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._global_instructions = self._load_global_instructions()

    def _load_global_instructions(self) -> list[str]:
        path = Path("config/global_instructions.json")
        if not path.exists():
            return []
        return json.loads(path.read_text())


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
            toolset = available_toolsets[ts.name]
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
        async def render_instruction(ctx: RunContext[deps_model]):

            rendered: list[str] = []

            # Get schema-defined placeholders
            schema_properties = self.deps.get("properties", {})
            schema_defaults = {
                name: prop.get("default")
                for name, prop in schema_properties.items()
                if "default" in prop
            }

            for instr in self._global_instructions + self.instructions:

                # Start with schema defaults
                context = dict(schema_defaults)

                # Override with runtime deps
                context.update(vars(ctx.deps))

                try:
                    rendered.append(instr.format(**context))
                except KeyError as e:
                    logger.error(
                        f"Skipping instruction '{instr}' - placeholder not found: {e}"
                    )
                    continue

            return "\n\n".join(rendered)

        return agent

