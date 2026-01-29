import asyncio
import copy
import yaml
import json
import inspect
import logging
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor, load_tokenizer
from slime.utils.types import Sample
from slime.utils.metric_utils import compute_pass_rate, compute_rollout_step, compute_statistics, dict_add_prefix

from slime.rollout.rm_hub import async_rm, batched_async_rm

from .code_metric import CodeMetricGatherer

__all__ = ["generate_rollout"]

logger = logging.getLogger(__name__)

class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


from enum import Enum
import re
from abc import ABC, abstractmethod
from typing import Optional, Any
from uuid import uuid4
from .tools import tool_registry, PythonSandbox
from .tool_parser import ToolParser, FunctionCall, extract_code_blocks_sync


class AgentData:
    """
    Encapsulates all state variables for the agent loop.
    
    AgentData is passed through state handlers and can be accessed by tools.
    This design follows verl's pattern for cleaner state management.
    
    Attributes:
        messages: Conversation history as list of message dicts
        sample: The Sample object being processed
        request_id: Unique identifier for this rollout request
        interaction: Optional interaction handler for multi-turn environments
        
        prompt_ids: Token ids for the initial prompt (fixed after PENDING state)
        response_ids: Token ids for all responses (LLM + observations)
        response_mask: Mask for response tokens (1=LLM generated, 0=observation/tool)
        response_logprobs: Log probabilities for response tokens
        
        turn_idx: Current turn index
        user_turns: Number of user/observation turns
        assistant_turns: Number of assistant response turns
        
        current_tool_calls: List of parsed tool calls from current response
        extra_fields: Dictionary for dynamic additions (e.g., tool session data)
    """
    
    def __init__(
        self,
        messages: list[dict[str, Any]],
        sample: 'Sample',
        request_id: str,
        interaction: Optional['BaseInteraction'] = None,
    ):
        self.messages = messages
        self.sample = sample
        self.request_id = request_id
        self.interaction = interaction
        
        # Token tracking state - explicitly separated
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []  # 1 for LLM tokens, 0 for observation tokens
        self.response_logprobs: list[float] = []
        
        # Turn counters
        self.turn_idx: int = 0
        self.user_turns: int = 0
        self.assistant_turns: int = 0
        
        # Tool call state
        self.current_tool_calls: list[FunctionCall] = []
        
        # Metrics and extra fields
        self.metrics: dict[str, Any] = {}
        self.extra_fields: dict[str, Any] = {}
    
    @property
    def total_turns(self) -> int:
        """Total number of conversation turns"""
        return self.user_turns + self.assistant_turns
    
    @property
    def total_response_length(self) -> int:
        """Total length of response tokens"""
        return len(self.response_ids)
    
    @property
    def effective_response_length(self) -> int:
        """Number of LLM-generated tokens (where mask=1)"""
        return sum(self.response_mask)
    
    def get_full_token_sequence(self) -> list[int]:
        """Get the complete token sequence (prompt + response)"""
        return self.prompt_ids + self.response_ids
    
    def get_full_loss_mask(self) -> list[int]:
        """Get the complete loss mask (0 for prompt, response_mask for response)"""
        return [0] * len(self.prompt_ids) + self.response_mask
    
    def get_full_log_probs(self) -> list[float]:
        """Get the complete log probs (0 for prompt, response_logprobs for response)"""
        return [0.0] * len(self.prompt_ids) + self.response_logprobs


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools" 
    INTERACTING = "interacting"
    TERMINATED = "terminated"

class BaseInteraction(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a session instance."""
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        Generates a response for the current turn of interaction.
        Returns:
        - should_terminate_sequence (bool)
        - response_content (str)
        - current_turn_score (float)
        - additional_data (dict)
        """
        return False, "", 0.0, {}

    async def calculate_score(self) -> float:
        """
        Calculates a score for the interaction,
        potentially considering aspects like partial exposure & in-context task switching.
        should be invoke at turn-level
        """
        return 0.0

    async def finalize_interaction(self) -> None:
        """
        Finalizes the interaction session and releases any associated state or resources.
        Simulates: release state
        """
        pass

class CodeInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.sandbox = PythonSandbox(
            timeout=config.get("timeout", 10),
            memory_limit=config.get("memory_limit", "100MB")
        )
        self.use_local_sandbox = config.get("local_run", False)
        self.sandbox_url = config.get("sandbox_url", None)
    
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        sample = kwargs.get("sample")
        
        # Extract the last assistant message which should contain code
        last_msg = messages[-1]
        
        # Simple extraction logic: check for code block
        code_blocks = re.findall(r"```(?:python\n)?(.*?)```", last_msg.get("content", ""), re.DOTALL)
        if not code_blocks:
            return False, "Please provide python code in a code block.", 0.0, {}
        
        code = code_blocks[-1].strip()

        # Get Ground Truth from sample if available
        ground_truth = {}
        if sample and sample.metadata:
            try:
                reward_model = sample.metadata.get('reward_model', {})
                if 'ground_truth' in reward_model:
                     ground_truth = json.loads(reward_model['ground_truth'])
            except Exception as e:
                logger.warning(f"Failed to load ground truth: {e}")

        # Execute code using PythonSandbox.execute_code
        output, status, meta = await self.sandbox.execute_code(
             sandbox_fusion_url=self.sandbox_url,
             memory_limit_mb=self.sandbox.memory_limit if isinstance(self.sandbox.memory_limit, int) else 1024,
             code=code,
             timeout=self.sandbox.timeout,
             language="python",
             ground_truth=ground_truth,
             local_run=self.use_local_sandbox
        )

        # Calculate Reward
        reward = await self.calculate_score(meta)
            
        return False, output, reward, meta

    async def calculate_score(self, meta: dict[str, Any]) -> float:
        """Calculate reward based on execution results"""
        pass_fail_list = meta.get("pass_fail_list", [])
        if not pass_fail_list:
            # If no test cases were run (syntax error or no GT), use status
            if meta.get("status") == "success":
                return 1.0
            elif meta.get("run_status") == "Error" or meta.get("exit_code", 0) != 0:
                return -1.0 # Significant penalty for runtime error
            return 0.0
        
        # All cases must pass for full reward
        if all(x == 1 for x in pass_fail_list):
            return 1.0
        
        return 0.0


class AgentLoop:
    """
    Manages the lifecycle of a single sample's rollout:
    Generate -> Tool Execution -> Observation -> Generate ...
    
    Refactored to use AgentData for state management and verl-style
    state handlers that return the next AgentState.
    """
    def __init__(self, 
                 args: Namespace, 
                 sample: Sample, 
                 sampling_params: dict[str, Any], 
                 state_manager: 'GenerateState',
                 interaction: Optional[BaseInteraction] = None,
                 tool_parser_name: str = "python_code"):

        self.args = args
        self.sample = sample
        self.sampling_params = sampling_params
        self.state_manager = state_manager
        self.interaction = interaction
        
        self.max_turns = getattr(args, "code_rollout_max_turn", 5)
        self.max_response_length = getattr(args, "rollout_max_response_len", 4096)
        
        # Extract execution config
        self.sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        
        # Initialize tool parser
        self.tool_parser = ToolParser.get_parser(tool_parser_name, state_manager.tokenizer)

    async def run(self) -> Sample:
        """
        Main entry point for the agent loop.
        
        Uses AgentData to encapsulate all state and runs the state machine
        until termination.
        """
        # Initialize AgentData with state from sample
        agent_data = AgentData(
            messages=list(self.sample.prompt),
            sample=self.sample,
            request_id=str(uuid4()),
            interaction=self.interaction,
        )
        
        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED
            
            # Safety condition: check max turns
            if agent_data.turn_idx >= self.max_turns and state != AgentState.TERMINATED:
                logger.info(f"Max turns ({self.max_turns}) reached. Terminating.")
                state = AgentState.TERMINATED
            
            # Safety condition: check max response length
            if agent_data.total_response_length >= self.max_response_length and state != AgentState.TERMINATED:
                logger.info(f"Max response length ({self.max_response_length}) reached. Terminating.")
                state = AgentState.TERMINATED
        
        # Finalize and return updated sample
        return self._finalize_sample(agent_data)

    async def _handle_pending_state(self, agent_data: AgentData) -> AgentState:
        """
        Initialize state and prepare prompt tokens.
        
        Returns:
            AgentState.GENERATING to start generation
        """
        # Tokenize the initial prompt
        prompt_ids = self.state_manager.tokenizer.apply_chat_template(
            agent_data.messages, tools=None, tokenize=True, add_generation_prompt=True
        )
        agent_data.prompt_ids = prompt_ids
        
        return AgentState.GENERATING

    async def _handle_generating_state(self, agent_data: AgentData) -> AgentState:
        """
        Generate tokens using SGLang and process the response.
        
        Returns:
            AgentState.PROCESSING_TOOLS if tool calls detected
            AgentState.INTERACTING if interaction configured
            AgentState.TERMINATED otherwise
        """
        # Apply template to get current prompt ids (including any new turns)
        current_prompt_ids = self.state_manager.tokenizer.apply_chat_template(
            agent_data.messages, tools=None, tokenize=True, add_generation_prompt=True
        )
        
        json_data = {
            "input_ids": current_prompt_ids,
            "sampling_params": self.sampling_params,
            "return_logprob": True,
            "logprob_start_len": max(0, len(current_prompt_ids) - 1)
        }

        # Async Request to SGLang
        try:
            response = await post(self.sglang_url, json_data)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            agent_data.sample.status = Sample.Status.FAILED
            return AgentState.TERMINATED

        response_text = response["text"]
        meta_info = response.get("meta_info", {})
        
        # Append assistant response to message history
        agent_data.messages.append({"role": "assistant", "content": response_text})
        agent_data.sample.response = response_text
        agent_data.assistant_turns += 1

        # Extract and track new tokens
        if "output_token_logprobs" in meta_info:
            new_tokens = [item[1] for item in meta_info["output_token_logprobs"]]
            new_logprobs = [item[0] for item in meta_info["output_token_logprobs"]]
        else:
            # Fallback if no logprobs returned
            new_tokens = self.state_manager.tokenizer.encode(response_text, add_special_tokens=False)
            new_logprobs = [0.0] * len(new_tokens)
        
        # Update AgentData with new response tokens (mask=1 for LLM generated)
        agent_data.response_ids.extend(new_tokens)
        agent_data.response_mask.extend([1] * len(new_tokens))
        agent_data.response_logprobs.extend(new_logprobs)

        # Handle other metadata
        agent_data.sample.update_from_meta_info(self.args, meta_info)

        # Extract tool calls using the parser
        _, tool_calls = await self.tool_parser.extract_tool_calls(response_text)
        
        if tool_calls:
            agent_data.current_tool_calls = tool_calls
            return AgentState.PROCESSING_TOOLS
        elif self.interaction:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """
        Execute detected tool calls and add observation to conversation.
        
        Returns:
            AgentState.GENERATING to continue conversation
            AgentState.TERMINATED if max turns reached
        """
        agent_data.turn_idx += 1
        agent_data.user_turns += 1
        
        logger.debug(f"Executing tool (turn {agent_data.turn_idx})")
        
        # Execute the last tool call (typically code_interpreter)
        if agent_data.current_tool_calls:
            tool_call = agent_data.current_tool_calls[-1]
            tool_args = tool_call.get_arguments_dict()
            
            result = await tool_registry.execute_tool(tool_call.name, tool_args)
        else:
            result = "No tool call to execute"
        
        # Format observation message
        observation_message = {
            "role": "user",
            "content": f"Execution Output:\n{result}"
        }
        agent_data.messages.append(observation_message)

        # Tokenize observation and add to response (mask=0 for observation)
        obs_tokens = self._tokenize_observation(observation_message)
        agent_data.response_ids.extend(obs_tokens)
        agent_data.response_mask.extend([0] * len(obs_tokens))
        agent_data.response_logprobs.extend([0.0] * len(obs_tokens))
        
        # Clear current tool calls
        agent_data.current_tool_calls = []
        
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """
        Handle interaction with environment/user.
        
        Returns:
            AgentState.GENERATING to continue
            AgentState.TERMINATED if interaction signals termination
        """
        if not agent_data.interaction:
            return AgentState.TERMINATED

        # Get response from interaction
        should_terminate, response_text, reward, meta = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, sample=agent_data.sample
        )
        agent_data.user_turns += 1
        
        if response_text:
            agent_data.messages.append({"role": "user", "content": response_text})
            
            # Tokenize interaction response (mask=0)
            obs_tokens = self._tokenize_observation({"role": "user", "content": response_text})
            agent_data.response_ids.extend(obs_tokens)
            agent_data.response_mask.extend([0] * len(obs_tokens))
            agent_data.response_logprobs.extend([0.0] * len(obs_tokens))

        # Handle reward
        if reward is not None:
            if agent_data.sample.reward is None:
                agent_data.sample.reward = reward
            elif isinstance(agent_data.sample.reward, (int, float)):
                agent_data.sample.reward += reward

        if should_terminate:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    def _tokenize_observation(self, message: dict[str, Any]) -> list[int]:
        """
        Tokenize an observation/user message for adding to response.
        
        Args:
            message: Message dict with 'role' and 'content'
            
        Returns:
            List of token ids
        """
        try:
            if hasattr(self.state_manager.tokenizer, "apply_chat_template"):
                formatted_text = self.state_manager.tokenizer.apply_chat_template(
                    [message],
                    tokenize=False,
                    add_generation_prompt=False
                )
                return self.state_manager.tokenizer.encode(formatted_text, add_special_tokens=False)
            else:
                return self.state_manager.tokenizer.encode(message["content"], add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Tokenization of observation failed: {e}. Fallback to content encoding.")
            return self.state_manager.tokenizer.encode(message["content"], add_special_tokens=False)

    def _finalize_sample(self, agent_data: AgentData) -> Sample:
        """
        Finalize the sample with data from AgentData.
        
        Args:
            agent_data: The AgentData containing accumulated state
            
        Returns:
            Updated Sample with tokens, masks, and metadata
        """
        sample = agent_data.sample
        
        # Set tokens and masks using AgentData helper methods
        sample.tokens = agent_data.get_full_token_sequence()
        sample.loss_mask = agent_data.get_full_loss_mask()
        sample.rollout_log_probs = agent_data.get_full_log_probs()
        sample.response_length = agent_data.total_response_length
        
        # Update conversation history
        sample.prompt = agent_data.messages
        
        # Set training metadata
        sample.train_metadata = {
            "_turn_idx": agent_data.turn_idx,
            "_user_turns": agent_data.user_turns,
            "_assistant_turns": agent_data.assistant_turns,
            "_effective_response_length": agent_data.effective_response_length,
        }
        
        return sample


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using the Agent Loop State Machine"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    
    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"
    
    # Initialize interaction if needed
    interaction_config = {
        "sandbox_url": getattr(args, "sandbox_url", None),
        "local_run": getattr(args, "sandbox_local_run", False),
        "timeout": getattr(args, "sandbox_default_time_limit_s", 10),
        "memory_limit": getattr(args, "sandbox_default_memory_limit_mb", 1024)
    }
    # For this example, we always use CodeInteraction
    interaction = CodeInteraction(interaction_config)

    # Initialize and run the Agent Loop
    agent_loop = AgentLoop(args, sample, sampling_params, state, interaction=interaction)
    return await agent_loop.run()



def get_feedback_msg(args, reward_cat):
    feedback_message_filepath = args.code_feedback_message_filepath
    version = args.code_feedback_message_version
    feedback_info = yaml.safe_load(open(feedback_message_filepath))[version]
    if reward_cat not in feedback_info:
        reward_cat = "default"
    return feedback_info[reward_cat]


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        # TODO Taro custom rollout, directly call generate func for code instead of custom generate
        with state.dp_rank_context() as _:
            sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        # assert sample.reward is not None, "code reward should be assigned in generate_fn"
        if sample.reward is None:
            sample.reward = await async_rm(args, sample)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_slime_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    await asyncio.gather(*[post(f"{url}/abort_request", {"abort_all": True}) for url in urls])

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset
    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = CodeMetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(f"First rollout sample: {sample=}")
                # logger.info(
                #     f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
                # )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                # code_execute_status_lst = [sample.reward['code_execute_status'] for sample in group]
                # metric_gatherer.log_code_execute_status(code_execute_status_lst)
                metric_gatherer.log_code_sample_train_metadata(group)
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {sample.label}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    metrics = {}
    for key in results.keys():
        metric_gatherer = CodeMetricGatherer()
        metric_gatherer.log_code_sample_train_metadata(results[key]["samples"])
        metrics |= dict_add_prefix(metric_gatherer.collect(add_prefix=False), f"eval/{key}/")
    return RolloutFnEvalOutput(data=results, metrics=metrics), []

async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sampling_params = base_sampling_params.copy()
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
