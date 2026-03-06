"""
PydanticAI todo agent.

This demonstrates a simple AI agent that manages a todo list using PydanticAI.
The agent uses tools to create, read, update, and delete todos.
"""

from pydantic_ai import Agent
from pydantic_ai.ag_ui import StateDeps
from dotenv import load_dotenv
from models import TodoState
from tools import tools
from copilot_model import CopilotModel

# Load environment variables (LOGFIRE_TOKEN, etc.)
load_dotenv()

# Create the agent
# - model: GitHub Copilot SDK model (uses the copilot CLI binary)
# - deps_type: The type of dependencies/state passed to tools (StateDeps wraps TodoState for AG-UI)
# - tools: Functions the agent can call to interact with the todo list
agent = Agent(
  model=CopilotModel(model_name_value="gpt-4.1"),
  deps_type=StateDeps[TodoState],
  tools=tools,
)
