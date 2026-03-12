from pydantic import BaseModel, Field
from pydantic_ai import UsageLimits, ModelSettings, ConcurrencyLimit

from typing import Literal, Any


class ChatskyToolset(BaseModel):
    name: str
    configuration: dict = Field(default_factory=dict)
    approval_required: list[str] = Field(default_factory=list)

class AdditionalConfiguration(BaseModel, arbitrary_types_allowed=True):
    usage_limits: UsageLimits
    toolset_management_policy: Literal["auto", "manual"]
    model_settings: ModelSettings
    concurrency_limit: ConcurrencyLimit
