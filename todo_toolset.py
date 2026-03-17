from pydantic_ai import FunctionToolset, Tool

class TodoToolset(FunctionToolset):

    def __init__(self, config: dict, approval_required: list[str]):
        self.max_tasks = config.get("max_tasks", 2)
        self.approval_required = approval_required

        self._tasks: list[str] = []

        super().__init__(tools=self._build_tools())

    # TODO: Think of a better way to organize this part
    def _build_tools(self) -> list[Tool]:
        return [
            Tool(self.create_task, requires_approval="create_task" in self.approval_required),
            Tool(self.list_tasks, requires_approval="list_tasks" in self.approval_required),
        ]

    async def create_task(self, title: str) -> str:
        if len(self._tasks) >= self.max_tasks:
            return "Max tasks reached"

        self._tasks.append(title)
        return f"Task created: {title}"

    async def list_tasks(self) -> list[str]:
        return list(self._tasks)
