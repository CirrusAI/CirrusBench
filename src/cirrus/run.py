import time
start_time = time.time()
from pathlib import Path
import requests
from loguru import logger
from cirrus.configs.paths import TASK_DIR

from dotenv import load_dotenv
from cirrus.utils.basic import (
    load_json_list,
)

from cirrus.utils.display import  Text

from cirrus.utils.basic_utils import OdysseyTask

from pydantic import BaseModel ,Field
from typing import Annotated , Optional
from cirrus.orchestrator.orchestrator import Orchestrator
from cirrus.configs.run_configs import RunConfig


def get_tasks_jsonl(fp):
    fp = Path(fp)
    tasks = load_json_list(fp)
    return tasks


def run_task(task:OdysseyTask,
             config:RunConfig = None,
             max_steps = 3,
             max_errors = 5):

    if max_steps <= 0:
        raise ValueError("Max steps must be greater than 0")
    if max_errors <= 0:
        raise ValueError("Max errors must be greater than 0")    
    


    logger.info(
        f"STARTING SIMULATION: Domain: {config.domain}, Task: {task.id},Model:{config.model_name}"
    )
    retry_times = 3
    seperate_indices = task.seperate_indices
    history_messages = task.messages


    orchestrator = Orchestrator(
            domain=config.domain,
            task=task,
            max_steps=max_steps,
            max_errors=max_errors,
            seed=6,
            validate_communication= False,
            bool_jump = False,
            bool_replace = False,
            model_name = config.model_name,
        )
    
    simulation = orchestrator.run()


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run benchmark simulation")
    parser.add_argument("--domain", type=str, default="no_tool", choices=["no_tool", "with_tool"], help="Task domain")
    parser.add_argument("--model-name", type=str, default="deepseek-chat", help="LLM model name")
    parser.add_argument("--task-ids", type=str, nargs="+", default=None, help="Task IDs to run, e.g. --task-ids 0008 0009")
    parser.add_argument("--num-tasks", type=int, default=None, help="Run only the first N tasks (mutually exclusive with --task-ids)")
    parser.add_argument("--num-trials", type=int, default=3, help="Number of trials per task")
    parser.add_argument("--max-steps", type=int, default=20, help="Max steps per trial")
    parser.add_argument("--max-errors", type=int, default=10, help="Max consecutive errors allowed")
    parser.add_argument("--max-concurrency", type=int, default=3, help="Max concurrent simulations")
    parser.add_argument("--save-to", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite existing results")
    parser.add_argument("--judge-model", type=str, default=None, help="Model for judge scoring (default: from configs/judge_config.yaml)")
    parser.add_argument("--judge-temperature", type=float, default=None, help="Temperature for judge scoring (default: from configs/judge_config.yaml)")
    return parser.parse_args()


def main():

    load_dotenv()

    args = parse_args()

    if args.task_ids and args.num_tasks is not None:
        raise ValueError("--task-ids and --num-tasks are mutually exclusive, specify only one")

    config = RunConfig(
        domain=args.domain,
        task_ids=args.task_ids,
        num_tasks=args.num_tasks,
        num_trials=args.num_trials,
        max_steps=args.max_steps,
        max_errors=args.max_errors,
        save_to=args.save_to,
        max_concurrency=args.max_concurrency,
        model_name=args.model_name,
        overwrite=args.overwrite,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
    )
    if config.task_set_name is None:
        task_set_name = config.domain
    else:
        task_set_name = config.task_set_name

    if config.domain == 'no_tool':    
        fp = TASK_DIR / 'en_tasks_without_tool.jsonl'
    elif config.domain == 'with_tool':
        fp = TASK_DIR / 'en_tasks_with_tool.jsonl'
    else:
        raise ValueError
    
    tasks = get_tasks_jsonl(fp)

    total_num_tasks = len(tasks)

    num_tasks = len(tasks)
    # console_text = Text(
    #     text=f"Running {num_tasks} out of {total_num_tasks} tasks for solo agent.",
    #     style="bold green",
    # )
    #ConsoleDisplay.console.print(console_text)

    if config.task_ids:
        selected_tasks = []
        for task in tasks:
            if task['id'] in config.task_ids:
                selected_tasks.append(task)
        if len(selected_tasks) == 0:
            raise ValueError
        tasks = selected_tasks[:]
    elif config.num_tasks is not None:
        tasks = tasks[:config.num_tasks]

    for task in tasks:


        task = OdysseyTask.model_validate(task)

        run_task(task,config)

if __name__ == '__main__':
    main()