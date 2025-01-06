import json
from typing import Any, Callable, Dict, List as TList

import tqdm

from synth import Task, Dataset, PBE, Example
from synth.syntax import (
    FunctionType,
    Program,
    UnknownType,
    guess_type,
)

from calculator import dsl, evaluator, FLOAT

dsl.instantiate_polymorphic_types(5)


def __convert__(load: Callable[[], Dataset[PBE]], name: str) -> None:
    tasks = load()
    tasks.save(name)
    sols = sum(1 for t in tasks if t.solution)
    print(f"Converted {len(tasks)} tasks {sols / len(tasks):.0%} containing solutions")
    # Integrity check
    for task in tqdm.tqdm(tasks, desc="integrity check"):
        for ex in task.specification.examples:
            obt = evaluator.eval(task.solution, ex.inputs)
            assert (
                obt == ex.output
            ), f"failed on {task.solution} inputs:{ex.inputs} got:{obt} target:{ex.output}"


def convert_calculator(
    file: str = "dataset/calculator_dataset.json",
    output_file: str = "calculator.pickle",
) -> None:
    def load() -> Dataset[PBE]:
        tasks: TList[Task[PBE]] = []
        with open(file, "r") as fd:
            raw_tasks: TList[Dict[str, Any]] = json.load(fd)
            for raw_task in tqdm.tqdm(raw_tasks, desc="converting"):
                name: str = raw_task["program"]
                raw_examples: TList[Dict[str, Any]] = raw_task["examples"]
                inputs = [raw_example["inputs"] for raw_example in raw_examples]
                outputs: TList = [raw_example["output"] for raw_example in raw_examples]
                args_types = [guess_type(arg) for arg in inputs[0]] + [
                    guess_type(outputs[0])
                ]
                # guess_type doesn't recognise FLOAT but since it is the only type not recognised we know that Unknown Type is acutally FLOAT
                args_types = [
                    at if not isinstance(at, UnknownType) else FLOAT
                    for at in args_types
                ]
                type_request = FunctionType(*args_types)
                prog: Program = dsl.parse_program(name, type_request)
                examples = [
                    Example(inp, out)
                    for inp, out in zip(inputs, outputs)
                    if out is not None
                ]
                if len(examples) < len(inputs):
                    continue
                tasks.append(
                    Task[PBE](type_request, PBE(examples), prog, {"name": name})
                )
        return Dataset(tasks, metadata={"dataset": "calculator", "source:": file})

    __convert__(load, output_file)


if __name__ == "__main__":
    import argparse

    argument_parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Convert calculator original dataset to ProgSynth format."
    )

    argument_default_values = {
        "output": "calculator.pickle",
    }

    argument_parser.add_argument(
        type=str,
        dest="file",
        action="store",
        help="Source JSON calculator file to be converted",
    )
    argument_parser.add_argument(
        "-o",
        "--output",
        type=str,
        action="store",
        default=argument_default_values["output"],
        help=f"Output dataset file in ProgSynth format (default: '{argument_default_values['output']}')",
    )
    parsed_parameters = argument_parser.parse_args()
    convert_calculator(parsed_parameters.file, parsed_parameters.output)
