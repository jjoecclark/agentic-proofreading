"""Test-only oracle: returns the answer stored next to it (answers.json, written by the test)."""
import json
import pathlib


def run(state, world, llm, workdir, limits):
    answers = json.loads((pathlib.Path(__file__).parent / "answers.json").read_text())
    return answers[state["task_id"]]
