"""Test-only: calls the LLM until the proxy refuses (budget), without catching the error."""


def run(state, world, llm, workdir, limits):
    for _ in range(1000):
        llm.messages.create(model=limits.models[0], max_tokens=20000,
                            messages=[{"role": "user", "content": "x" * 2000}])
