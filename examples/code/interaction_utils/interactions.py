import re
from abc import ABC
from typing import Optional, Any
from uuid import uuid4
from examples.code.tool_utils.tools import PythonSandbox
from examples.code.interaction_utils.interaction_registry import InteractionRegistry
import json
import logging
from typing import Any
from examples.code.global_utils import get_event_loop



logger = logging.getLogger(__name__)


class BaseInteraction(ABC):
    # Default role for interaction responses (can be 'user' or 'tool')
    response_role: str = "tool"
    _class_initialized: bool = False
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.name: str = config.get("name", "interaction_agent")
        # Allow role to be overridden from config
        self.response_role = config.get("response_role", self.__class__.response_role)
        # Call class-level initialization (only runs once)
        self._init_class(config)
    
    @classmethod
    def _init_class(cls, config: dict[str, Any]):
        """Class-level initialization that only runs once.
        
        Override this in subclasses to do heavy initialization work that should
        be shared across all instances (like creating Ray Actors).
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True


    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a session instance."""
        if instance_id is None:
            return str(uuid4())
        else:
            return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, dict[str, Any], dict[str, Any]]:
        """
        Generates a response for the current turn of interaction.
        Returns:
        - should_terminate_sequence (bool)
        - response_content (str)
        - reward_result (dict): Matches RewardFn format with keys:
            - reward_value (float): The actual reward value
            - score (int): 0 or 1 for pass/fail
            - reward_cat (str): Category like 'accept', 'wrong_answer', 'no_code', etc.
            - extra_info (CodeExtraInfo): Execution metadata
        - additional_data (dict)
        """
        return False, "", {}, {}

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

@InteractionRegistry.register("code")
class CodeInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.config = config
        self.loop = get_event_loop()

    @classmethod
    def _init_class(cls, config: dict[str, Any]):
        """Initialize class-level shared resources (only runs once)."""
        if cls._class_initialized:
            return
        cls.sandbox = PythonSandbox(
            **config["tool_config"]
        )
        cls.use_local_sandbox = config.get("use_local_sandbox", False)
        cls .response_role: str = "tool"
    
    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        import time
        from examples.code.code_metric import CodeExtraInfo
        
        start_time = time.monotonic()
        sample = kwargs.get("sample")
        extra_info = CodeExtraInfo()
        # Extract the last assistant message which should contain code
        last_msg = messages[-1]
        
        # Simple extraction logic: check for code block
        code_blocks = re.findall(r"```(?:python\n)?(.*?)```", last_msg.get("content", ""), re.DOTALL)
        if not code_blocks:
            extra_info.no_code_extract = True
            extra_info.code_reward_time = round(time.monotonic() - start_time, 4)
            reward_result = dict(
                reward_value=0.0,
                score=0,
                reward_cat="no_code",
                extra_info=extra_info,
            )
            return False, "Please provide python code in a code block.", reward_result, {}
        
        code = code_blocks[-1].strip()
        extra_info.no_code_extract = False
        extra_info.extracted_code = code

        # Get Ground Truth from sample if available
        ground_truth = {}
        time_limit = self.sandbox.timeout
        memory_limit_mb = self.sandbox.memory_limit if isinstance(self.sandbox.memory_limit, int) else 1024
        
        if sample and sample.metadata:
            try:
                reward_model = sample.metadata.get('reward_model', {})
                if 'ground_truth' in reward_model:
                     ground_truth = json.loads(reward_model['ground_truth'])
                # Use custom time/memory limits if available
                if reward_model.get("time_limit"):
                    time_limit = reward_model["time_limit"]
                if reward_model.get("memory_limit_mb"):
                    memory_limit_mb = reward_model["memory_limit_mb"]
            except Exception as e:
                logger.warning(f"Failed to load ground truth: {e}")

        # Execute code using PythonSandbox.execute_code
        try:
            output, status, meta = await self.sandbox.execute(
                memory_limit_mb=memory_limit_mb,
                code=code,
                timeout=time_limit,
                language="python",
                ground_truth=ground_truth,
                local_run=self.use_local_sandbox
            )
            extra_info.meta_data = meta
            
            # Track execution time
            duration_lst = meta.get("duration", None)
            if duration_lst is not None:
                if not isinstance(duration_lst, list):
                    duration_lst = [duration_lst]
                extra_info.code_execute_time = sum(duration_lst) / len(duration_lst)
                extra_info.code_execute_time_max = max(duration_lst)

            # Calculate Reward using aligned logic
            reward_result = await self.calculate_score(meta, extra_info, status)
            extra_info.code_reward_time = round(time.monotonic() - start_time, 4)

        except Exception as e:
            logger.warning(f"Error in CodeInteraction: {e}")
            extra_info.code_reward_error = True
            extra_info.code_reward_time = round(time.monotonic() - start_time, 4)
            reward_result = dict(
                reward_value=0.0,
                score=0,
                reward_cat="python_error",
                extra_info=extra_info,
                error_details=str(e)
            )
            return False, str(e), reward_result, {}

            
        return reward_result["reward_value"] == 1.0, output, reward_result, meta

    async def calculate_score(self, meta: dict[str, Any], extra_info: "CodeExtraInfo", exec_status: str) -> float:
        pass_rate = 0.0
        score = 0
        answer_reward = 0.0
        pass_fail_list = meta.get("pass_fail_list", [])
        if pass_fail_list:
            extra_info.pass_fail_list = pass_fail_list
        
        stdout = meta.get("stdout", "")
        match_test_pass_rate = re.search(r"pass rate: \*\*(.*?)\*\*", stdout)
        if match_test_pass_rate:
            pass_rate = float(match_test_pass_rate.group(1))
        else:
            pass_fail_list = meta.get("pass_fail_list", [])
            if pass_fail_list:
                pass_rate = sum(pass_fail_list) / len(pass_fail_list)
            elif meta.get("api_status").lower() == "success":
                pass_rate = 1.0
        
        extra_info.pass_rate = pass_rate
        
        if pass_rate == 1.0:
            score = 1
            answer_reward = 1.0
            reward_cat = "accept"
        elif meta.get("run_status").lower() == "error" or meta.get("exit_code", 0) != 0:
            reward_cat = "runtime_error"
        else:
            reward_cat = "wrong_answer"
        
        return dict(
            reward_value=answer_reward,
            score=score,
            reward_cat=reward_cat,
            extra_info=extra_info,
        )