from pathlib import Path

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama


# Locate the main fair_fairness_benchmark repository folder. 
REPO_ROOT = Path(__file__).resolve().parents[1]


@tool
def list_reproduction_scripts() -> str:
    """List the available Python reproduction scripts in the FFB repository."""
    reproduce_folder = REPO_ROOT / "reproduce"
    scripts = sorted(file.name for file in reproduce_folder.glob("*.py"))

    print("\n[TOOL CALLED] list_reproduction_scripts")

    if not scripts:
        return "No reproduction scripts were found."

    return "\n".join(scripts)


# Connects the agent to the Qwen model running locally through Ollama.
model = ChatOllama(
    model="qwen3:4b",
    temperature=0,
)

# Give the agent its model, tool, and instructions.
agent = create_deep_agent(
    model=model,
    tools=[list_reproduction_scripts],
    system_prompt=(
        "You are an assistant for the Fair Fairness Benchmark repository. "
        "When asked about available reproduction experiments, you must call "
        "the list_reproduction_scripts tool before answering."
    ),
)

# Give the agent a task. 
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the repository tool and tell me which reproduction "
                    "experiments are available."
                ),
            }
        ]
    }
)

print("\nFINAL ANSWER:")
print(result["messages"][-1].content)