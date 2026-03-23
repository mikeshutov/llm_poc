from agents.synthesis.prompts.synthesis_schema_prompt import SYNTHESIS_SCHEMA


def build_solver_prompt(*, plan_block: str, task: str) -> str:
    return (
        f"Solve the following task or problem. To solve the problem, we have made step-by-step Plan and retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might contain irrelevant information."
        f"Plan with Evidence: {plan_block}"
        f"Now solve the question or task according to provided Evidence above."
        f"{SYNTHESIS_SCHEMA}"
        f"Task: {task}"
    )
