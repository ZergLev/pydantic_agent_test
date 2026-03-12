from pydantic_ai import FunctionToolset

class TaskStore:
    def __init__(self):
        self.tasks = []
        self.calls = []

store = TaskStore()


def create_task(title: str) -> str:
    """Create a task."""
    store.calls.append("create_task")
    store.tasks.append(title)
    return f"Task created: {title}"


def list_tasks() -> list[str]:
    """List all tasks."""
    store.calls.append("list_tasks")
    return store.tasks


some_toolset = FunctionToolset(
    tools=[create_task, list_tasks]
)
