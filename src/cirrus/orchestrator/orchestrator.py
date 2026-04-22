import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional , Literal
from pathlib import Path
from loguru import logger

from cirrus.agent.base import AgentError, BaseAgent, is_valid_agent_history_message
from cirrus.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from cirrus.data_model.simulation import SimulationRun, TerminationReason
from cirrus.data_model.tasks import EnvFunctionCall, InitializationData, Task
from cirrus.environment.environment import Environment, EnvironmentInfo
from cirrus.utils.utils import format_time, get_now
import random
import requests

from cirrus.configs.paths import TASK_POLICY_PROMPT_PATH ,REFERENCE_DIR

from cirrus.utils.basic import dump_json_list,load_json_list,load_json_dict
import json
from cirrus.llm import call_llm
from cirrus.judge.scoring import scoring_content
from cirrus.configs.run_configs import RunConfig
from cirrus.llm.generate import generate
from cirrus.utils.basic_utils import OdysseyTask

class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"




class Orchestrator:
    """
    Orchestrator for the simulation given a task.
    Passes messages between the Agent, User, and Environment.

    Communication Protocol:
        The orchestrator manages message flow between three roles: AGENT, USER, and ENV(ironment).
        Messages are passed in a turn-based manner following these rules:

        Message Types:
            - AssistantMessage: Sent by the agent
            - UserMessage: Sent by the user
            - ToolMessage: Sent by the environment in response to tool calls
            - MultiToolMessage: Wraps multiple tool messages when multiple tool calls are made

        Message Content Rules:
            1. Messages must contain EITHER text content OR tool calls, never both
            2. Messages cannot be empty (must have either text or tool calls)
            3. Tool calls must be followed by corresponding tool messages from the environment

        Communication Flow:
            - AGENT -> USER: Agent sends text response to user
            - AGENT -> ENV: Agent makes tool call(s) to environment
            - USER -> AGENT: User sends text message to agent
            - ENV -> AGENT: Environment returns tool results to agent (after agent's tool call)

    """

    def __init__(
        self,
        domain: Literal['with_tool','no_tool'] ,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        validate_communication: bool = False,
        bool_jump = False,
        bool_replace = False,
        model_name:str = 'gpt-4o-0806',
        overwrite = False
    ):
        """
        Initialize the Orchestrator for managing simulation between Agent, User, and Environment.
        """

        self.domain = domain
        self.task = task
        self.seed = seed
        self.validate_communication = validate_communication
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[Any] = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors = 0
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None
        self.bool_jump =  False
        self.bool_replace = False
        self.model_name = model_name
        self.overwrite = overwrite


    def get_policy(self):

        with open(TASK_POLICY_PROMPT_PATH, "r") as fp:
            policy = fp.read()
        return policy
        

    def get_references(self):
        if self.domain == 'with_tool':
            ref_fd = REFERENCE_DIR / 'references_withtool'
        elif self.domain == 'no_tool':
            ref_fd =  REFERENCE_DIR / 'references_notool'

        ref_fp = ref_fd / f'references_{self.task.id}.json'
        try:
            ref = load_json_dict(ref_fp)
        except (ValueError, json.JSONDecodeError):
            ref = []
        return ref


    def get_agent_tool_call_indices(self):
        indices = []
        for item in self.task.seperate_indices:
            indices += item['agent_tool_indices']
        return indices

    def get_agent_tool_call_messages(self):
        tool_mock = []
        known_info_list = []
        for msg_index ,msg in enumerate(self.task.messages):

            if msg_index in self.agent_tool_call_indices:
                known_info = {}
                tool_name = msg['tool_calls'][0]['function']['name']
                tool_name = tool_name.replace('#','_')
                arguments = json.loads(msg['tool_calls'][0]['function']['arguments'])
                tool_mock.append({'tool_name':tool_name,
                                  'tool_call':msg['tool_calls'][0],
                                  'result':self.task.messages[msg_index+1]['content']})
                for key in arguments:
                    known_info[key] = arguments[key]
                known_info_list.append(known_info)
        self.known_info = known_info_list
        return tool_mock

    def get_tool_schema(self):
    
        if not self.tool_mock:
            return []
        tool_schemas = []
        tool_schemas_all = load_json_list('./data/tool_schema_v2.jsonl')
        for item in self.tool_mock:
            tool_name  = item['tool_name']
            for schema in tool_schemas_all:
                if schema['function']['name'] == tool_name:
                    tool_schemas.append(schema)
        return tool_schemas

    def check_tool_call(self,test_tool_call):
        for tool_mock_item in self.tool_mock:
            if tool_mock_item['tool_name'] == test_tool_call['function']['name']:
                gt_tool_call = tool_mock_item['tool_call']
                gt_args = json.loads( gt_tool_call['function']['arguments'])
                tool_state = True
                test_tool_call_args = json.loads(test_tool_call['function']['arguments'])
                for key,value in test_tool_call_args.items():
                    if gt_args.get(key,None) == value:
                        pass
                    else:
                        tool_state = False
                if tool_state:
                    return tool_mock_item['result'],tool_state
        return 'Tool call mismatch', False

    def get_result_filepath(self):
        if self.domain == 'no_tool':
            result_fd = Path(f'./outputs/simulations/results_notool/{self.model_name}')
        elif self.domain == 'with_tool':
            result_fd = Path(f'./outputs/simulations/results_withtool/{self.model_name}')
        result_fd.mkdir(exist_ok=True,parents=True)
        result_fp_name =  f"result_{self.task.id}.json"
        result_fp = result_fd/result_fp_name

        return result_fp


    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        if self.domain not in ['with_tool','no_tool']:
            raise ValueError
        self.from_role = Role.USER
        self.to_role = Role.AGENT
        self.message_index = 0
        message = self.task.messages[self.message_index]
        self.message = UserMessage.model_validate(message)
        self.trajectory = [self.message]
        self.message_index += 1
        self.sub_task_id = 0
        self.sub_task_indices = [item['reply_index'] for item in self.task.seperate_indices]
        self.agent_tool_call_indices = self.get_agent_tool_call_indices()
        self.tool_mock = self.get_agent_tool_call_messages()
        self.result = {}
        self.result['subtasks'] = []
        self.result['task'] = self.task.model_dump()
        self.result['tool_calls'] = []
        self.result['tool_mocks'] = self.tool_mock
        self.stop = False
        self.policy = self.get_policy()


        references = self.get_references()
        self.tool_schema = self.get_tool_schema()
        information_retrieval = f'''
        <information_retrieval>
        Prioritize autonomous reasoning using provided references and tools before soliciting user input. Minimize unnecessary clarification cycles.
        <help_doc>
        {json.dumps(references,indent=2,ensure_ascii = False)}
        </help_doc>
        </information_retrieval>'''
        system_prompt = self.policy + information_retrieval + f'\n<known_info>{json.dumps(self.known_info,indent=2,ensure_ascii=False)}\n</known_info>'
        
        self.system_message = SystemMessage(content=system_prompt,role='system')




    def run(self):
        """
        Run the simulation.

        Returns:
            SimulationRun: The simulation run.
        """
        start_time = get_now()
        start = time.perf_counter()
        self.message_index = 0
        self.sub_task_id = 0
        logger.info(f'Starting task: {self.task.id}')
        task_messages = self.task.messages
        self.initialize()

        result_fp = self.get_result_filepath()
        if result_fp.exists() and self.overwrite:
            logger.info('Result already exists for this task, skipping')
            return
        while self.message_index < len(task_messages):

            self.step()
            if self.stop:
                break
        self.save()
        return

    def save(self):
        self.result['trajectory'] = [msg.model_dump() for msg in self.trajectory]
        result_fp = self.get_result_filepath()
        with open(result_fp,'w',encoding='utf-8') as f:
            json.dump(self.result,f,indent=4,ensure_ascii=False)

    def step(self):
        """
        Perform one step of the simulation.
        Sends self.message from self.from_role to self.to_role
        This can either be a message from agent to user/environment, environment to agent, or user to agent
        Updates self.trajectory
        """
        if self.done:
            raise ValueError("Simulation is done")

        # logger.debug(
        #     f"\nStep {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        # )
        # logger.debug(
        #     f"\nStep {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: \n{self.message}"
        # )

        # AGENT/ENV -> USER
        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            user_msg = UserMessage.model_validate(self.task.messages[self.message_index])
            user_msg.validate()
            self.trajectory.append(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            if user_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.AGENT
        # USER/ENV -> AGENT
        elif (
            self.from_role == Role.USER or self.from_role == Role.ENV
        ) and self.to_role == Role.AGENT:
            if self.message_index in self.sub_task_indices:
                ###进行子任务
                self.sub_task_id += 1
                scores = []      #记录每次trial的分数
                sub_task_result = []  #记录每次模拟的记录
                sub_task_state = False
                ### 每个子任务尝试3次
                for trial in range(2):
                    ### 进行一次trial
                    sub_task_trial = {"trial":trial+1}  ## 用这个trial 记录试验

                    sub_task_messages = []  ## 本次试验生成的所有消息
                    messages = deepcopy(self.trajectory)   ### 所有的历史消息

                    ## 尝试生成回复
                    #agent_msg,success = generate_by_local_Qwen(model=self.model_name,messages =[self.system_message]+messages+sub_task_messages,test = False)
                    # print(self.system_message)
                    # print(messages)
                    # print(sub_task_messages)
                    agent_msg,success = generate(model=self.model_name,messages =[self.system_message]+messages+sub_task_messages,tools=self.tool_schema)
                    #agent_msg,success = generate_by_LLMService(model=self.model_name,messages =[self.system_message]+messages+sub_task_messages,tools=self.tool_schema)
                    
                    if success:
                        pass
                    else:
                        sub_task_trial['sub_task_messages'] = []
                        sub_task_trial['score'] = 0
                        scores.append(0)
                        sub_task_result.append(sub_task_trial)
                        logger.debug(f'Trial {trial+1} failed, score=0')
                        sub_task_state = False
                        continue

                    tool_trial_nums = 0
                    ## 判断agent msg有没有tool call
                    if agent_msg.is_tool_call():
                        print(agent_msg.tool_calls)
                        while agent_msg.is_tool_call():
                            tool_trial_nums += 1
                            sub_task_messages.append(agent_msg)
                            tool_content ,tool_state = self.check_tool_call(agent_msg.tool_calls[0])
                            tool_msg = ToolMessage(role='tool',tool_call_id=agent_msg.tool_calls[0].get('id','-1'),content=tool_content)
                            self.result['tool_calls'].append({'tool_call':agent_msg.model_dump(),'result':tool_msg.model_dump()})
                            if tool_state:
                                tool_trial_nums = 0 

                            sub_task_messages.append(tool_msg)
                            
                            if tool_trial_nums >3:
                                break

                            #messages.append(agent_msg)
                            agent_msg,success = generate(model=self.model_name,messages =[self.system_message]+messages+sub_task_messages , tools=self.tool_schema)
                            #agent_msg,success = generate_by_LLMService(model=self.model_name,messages =[self.system_message]+messages+sub_task_messages , tools=self.tool_schema )
                            
                            if not success:
                                break
                    
                    if success:
                        pass
                    else:
                        sub_task_trial['sub_task_messages'] = []
                        sub_task_trial['score'] = 0
                        scores.append(0)
                        sub_task_result.append(sub_task_trial)
                        logger.debug(f'Trial {trial+1} failed, score=0')
                        sub_task_state = False
                        continue

                    sub_task_messages.append(agent_msg)
                    agent_msg.validate()
                    #score = simulate_scoring()
                    history = [msg.model_dump() for msg in self.trajectory]


                    score = scoring_content(contentA=self.task.messages[self.message_index]['content'],
                                        contentB= agent_msg.content,
                                        history=json.dumps(history,indent=2,ensure_ascii=False))
                    scores.append(score)        
                    logger.info(f'Subtask {self.sub_task_id}, trial {trial+1}, score: {score:1d}')
                    sub_task_trial['sub_task_messages'] = [msg.model_dump() for msg in sub_task_messages]
                    sub_task_trial['score'] = score
                    sub_task_result.append(sub_task_trial)
                    if score > 0.8:
                        logger.info(f'Subtask {self.sub_task_id} passed')
                        sub_task_state = True
                        break
                self.trajectory.extend(sub_task_messages)
                self.message = agent_msg

                self.result['subtasks'].append({'subtask_id':self.sub_task_id,
                                                'pass':sub_task_state,
                                                'score':max(scores),
                                                'simulation':sub_task_result,
                                                })
                if not sub_task_state:
                    self.stop = True
                # if success:
                #     logger.debug(f"\nStep {self.step_count}. 模型生成回复: \n{self.message}")

     #     self.message, self.agent_state
                # )
                # agent_msg.validate()
            elif self.message_index in self.agent_tool_call_indices:
                self.message_index +=2
                return

            else:
                agent_msg = AssistantMessage.model_validate(self.task.messages[self.message_index])
                self.message = agent_msg
                self.trajectory.append(agent_msg)

            if self.stop:
                return
            
            self.message = agent_msg
            self.from_role = Role.AGENT
            if agent_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.USER

        # AGENT/USER -> ENV
        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role ==  Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            tool_msgs = []
            message =  self.task.messages[self.message_index]
            self.message = ToolMessage.model_validate(message)
            # for tool_call in self.message.tool_calls:
            #     tool_msg = self.environment.get_response(tool_call)
            #     if tool_msg.error:
            #         self.num_errors += 1
            #     tool_msgs.append(tool_msg)
            # assert len(self.message.tool_calls) == len(
            #     tool_msgs
            # ), "Number of tool calls and tool messages should be the same"
            self.trajectory.append(self.message)
            self.to_role = self.from_role
            self.from_role = Role.ENV
        else:
            raise ValueError(
                f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}"
            )

        self.message_index += 1
        
        if self.validate_communication:
            self.check_communication_error()
        self.step_count += 1


    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        trajectory = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    num_expected_tool_messages = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages == 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ):
        """
        Initialize the environment.
        """
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

    def _get_environment_info(self) -> EnvironmentInfo:
        """
        Get the environment info.
        """
        return self.environment.get_info()

    def _count_errors(self, message_history: list[Message]) -> int:
        """
        Count the number of errors in the message history.
        """
        return sum(
            1 for msg in message_history if isinstance(msg, ToolMessage) and msg.error
        )

    def _add_timestamps(
        self, message_history: list[Message]
    ) -> list[tuple[str, Message]]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history


