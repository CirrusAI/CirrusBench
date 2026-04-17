
from copy import deepcopy
import time
start_time = time.time()


from pydantic import BaseModel ,Field
from typing import Annotated , Optional
from cirrus.configs.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SAVE_TO,
    DEFAULT_SEED,)
from cirrus.judge.scoring import _load_judge_config


class RunConfig(BaseModel):
    domain: Annotated[
        str,
        Field(
            description="The domain to run the simulation on",
            default="odyssey",
        ),
    ]
    model_name:Annotated[
        Optional[str],
        Field(
            description="The task set to run the simulation on. If not provided, will load default task set for the domain.",
            default='qwen3-max',
        ),
    ]

    task_set_name: Annotated[
        Optional[str],
        Field(
            description="The task set to run the simulation on. If not provided, will load default task set for the domain.",
            default=None,
        ),
    ]
    task_split_name: Annotated[
        Optional[str],
        Field(
            description="The task split to run the simulation on. If not provided, will load 'base' split.",
            default="base",
        ),
    ]
    task_ids: Annotated[
        Optional[list[str]],
        Field(
            description="The task IDs to run the simulation on",
            default=None,
        ),
    ]
    num_tasks: Annotated[
        Optional[int],
        Field(
            description="The number of tasks to run the simulation on",
            default=None,
        ),
    ]
    is_remote: Annotated[
        bool,
        Field(
            description="Whether to run the simulation remotely",
            default=False,
        ),
    ]
    agent: Annotated[
        str,
        Field(
            description="The type of agent to run the simulation on",
            default="llm_agent",
        ),
    ]
    llm_agent: Annotated[
        str,
        Field(
            description="The model to use for the agent",
            default=DEFAULT_LLM_AGENT,
        ),
    ]
    llm_args_agent: Annotated[
        dict,
        Field(
            description="The arguments to pass to the LLM for the agent",
            default_factory=lambda: deepcopy(DEFAULT_LLM_ARGS_AGENT),
        ),
    ]
    user: Annotated[
        str,
        Field(
            description="The type of user to run the simulation on",
            default="user_simulator",
        ),
    ]
    llm_user: Annotated[
        str,
        Field(
            description="The model to use for the user",
            default=DEFAULT_LLM_USER,
        ),
    ]
    llm_args_user: Annotated[
        dict,
        Field(
            description="The arguments to pass to the LLM for the user",
            default_factory=lambda: deepcopy(DEFAULT_LLM_ARGS_USER),
        ),
    ]
    num_trials: Annotated[
        int,
        Field(
            description="The number of trials to run the simulation on",
            default=DEFAULT_NUM_TRIALS,
        ),
    ]
    max_steps: Annotated[
        int,
        Field(
            description="The maximum number of steps to run the simulation",
            default=DEFAULT_MAX_STEPS,
        ),
    ]
    max_errors: Annotated[
        int,
        Field(
            description="The maximum number of tool errors allowed in a row in the simulation",
            default=DEFAULT_MAX_ERRORS,
        ),
    ]
    save_to: Annotated[
        Optional[str],
        Field(
            description="The path to json file where to save the simulation results",
            default=DEFAULT_SAVE_TO,
        ),
    ]
    max_concurrency: Annotated[
        int,
        Field(
            description="The maximum number of concurrent simulations to run",
            default=DEFAULT_MAX_CONCURRENCY,
        ),
    ]
    seed: Annotated[
        Optional[int],
        Field(
            description="The seed to use for the simulation",
            default=DEFAULT_SEED,
        ),
    ]
    log_level: Annotated[
        Optional[str],
        Field(
            description="The log level to use for the simulation",
            default=DEFAULT_LOG_LEVEL,
        ),
    ]
    enforce_communication_protocol: Annotated[
        bool,
        Field(
            description="Whether to enforce communication protocol rules (e.g., no mixed messages with text and tool calls)",
            default=False,
        ),
    ]
    overwrite : Annotated[
        bool,
        Field(
            description="weather overwrite existed results",
            default=False,
        ),
        ]
    judge_model: Annotated[
        Optional[str],
        Field(
            description="The model to use for judge scoring. If not provided, uses configs/judge_config.yaml default.",
            default=None,
        ),
    ]
    judge_temperature: Annotated[
        Optional[float],
        Field(
            description="The temperature for judge scoring. If not provided, uses configs/judge_config.yaml default.",
            default=None,
        ),
    ]

    def get_judge_config(self) -> dict:
        """Return effective judge config, falling back to judge_config.yaml for unset fields."""
        yaml_cfg = _load_judge_config()
        return {
            "model": self.judge_model if self.judge_model is not None else yaml_cfg.get("model", "deepseek-chat"),
            "temperature": self.judge_temperature if self.judge_temperature is not None else yaml_cfg.get("temperature", 0.0),
        }

    def validate(self) -> None:
        """
        Validate the run config
        """
        pass