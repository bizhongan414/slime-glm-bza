import asyncio
import logging
import re
import json
import time
from types import SimpleNamespace

from examples.code.sandbox_utils import execute_code

# logger = logging.getLogger(__name__)
from loguru import logger
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""), 
    colorize=True, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO"
)
logging.getLogger().setLevel(logging.INFO)

_TIMEOUT = 60

class RewardFn:
    def __init__(self, args):
        self.default_time_limit_s = args.sandbox_default_time_limit_s
        self.default_memory_limit_mb = args.sandbox_default_memory_limit_mb
        self.local_run = args.sandbox_local_run

        self.use_case_custom_time_limit = args.sandbox_use_case_custom_time_limit
        self.use_case_custom_memory_limit = args.sandbox_use_case_custom_memory_limit
        
        self.max_turns = args.code_rollout_max_turn
        # print(f"[taro_debug] reward fn config:", self.__dict__)
        

    async def __call__(self, sample, **kwargs):
        code_execute_status = {}
        start_time = time.monotonic()
        try:
            answer_reward = 0.0
            format_reward = 0.0
            timeout_reward = 0.0

            extracted_code = _extract_code_from_answer(sample.response)
            if extracted_code is not None:
                code_execute_status["no_code_extract"] = 0
                
                ground_truth = json.loads(sample.metadata['reward_model']['ground_truth'])
                
                time_limit = self.use_case_custom_time_limit and ground_truth.get("time_limit", None) or self.default_time_limit_s
                memory_limit_mb = self.use_case_custom_memory_limit and ground_truth.get("memory_limit_mb", None) or self.default_memory_limit_mb
                if self.local_run:
                    time_limit += 5

                response, exec_statu, meta_data = execute_code(
                    sandbox_fusion_url=None,
                    memory_limit_mb=memory_limit_mb,
                    ground_truth=ground_truth,
                    code=extracted_code,
                    timeout=time_limit,
                    language="python",
                    local_run=self.local_run
                )
                
                # logger.info(f"[taro_debug] meta_data: {repr(meta_data)}")
                duration_lst = meta_data.get("duration", None)
                if duration_lst is not None:
                    if not isinstance(duration_lst, list):
                        duration_lst = [duration_lst]
                    code_execute_status["code_execute_time"] = sum(duration_lst) / len(duration_lst)
                    code_execute_status["code_execute_time_max"] = max(duration_lst)
                    
                if exec_statu.lower() == "timeout":
                    logger.warning("execute code timeout, not implement")
                    reward_msg = "exec_timeout"
                else:
                    match_test_pass_rate = re.search(r"pass rate: \*\*(.*?)\*\*", meta_data["stdout"])
                    pass_rate = float(match_test_pass_rate.group(1)) if match_test_pass_rate else 1.0
                    logger.info(f"{pass_rate=}")
                    answer_reward = 1
                    reward_msg = "success"
            else:
                reward_msg = "no_code"
                code_execute_status["no_code_extract"] = 1
            # reward = answer_reward + format_reward + timeout_reward
            reward = answer_reward
            code_execute_status["code_reward_time"] = round(time.monotonic() - start_time, 4)
            return dict(
                reward_value=reward,
                reward_cat=reward_msg,
                code_execute_status=code_execute_status,
                extracted_code=extracted_code
            )
        except Exception as e:
            logger.warning(f"Error in RewardFn: {e=} {sample.prompt=} {sample.response=}")
            code_execute_status['code_reward_error'] = 1
            code_execute_status["code_reward_time"] = round(time.monotonic() - start_time, 4)
            return dict(
                reward_value=0.0, 
                reward_cat="python_error",
                code_execute_status=code_execute_status,
                error_details=str(e)
            )

def _extract_code_from_answer(solution_text: str) -> str | None:
    try:
        think_end_tag = "</think>"
        if think_end_tag in solution_text:
            search_start_index = solution_text.rindex(think_end_tag) + len(think_end_tag)
            solution_text = solution_text[search_start_index:]
        code_match = re.search(r"```(?:python\n)?(.*?)```", solution_text, re.DOTALL)
        return code_match.group(1).strip() if code_match else None
    except ValueError:
        return None

# _REWARD_FN: RewardFn | None = None

async def reward_fn(args, sample, **kwargs):
    # global _REWARD_FN
    # if _REWARD_FN is None:
    #     _REWARD_FN = RewardFn(args)
    _REWARD_FN = RewardFn(args)
    return await _REWARD_FN(sample, **kwargs)

def postprocess_reward(output):
    score = output['reward_value']
    return {
        "score": score,
        "acc": score,
        "pred": None,
    }

def cut_gts(ground_truth, size):
    if size >= len(ground_truth["inputs"]):
        return ground_truth
    return {
        "inputs": ground_truth["inputs"][:size],
        "outputs": ground_truth["outputs"][:size],
    }


if __name__ == "__main__":
    # Run this UT with:
    # python -m examples.code.single_turn_reward_fn
    test_file = "/afs/chatrl/users/lyy/code/slime/examples/code/test_reward_fn_data.json"
    import ray
    ray.init()
    with open(test_file) as f:
        for idx, line in enumerate(f):
            # if idx in [0]:
            #     continue
            data = json.loads(line)
            test_prompt = data['input']
            test_response = data['output']
            # print(test_response)
            reward_model = data['reward_model']
            ground_truth = json.loads(reward_model['ground_truth'])
            ground_truth = cut_gts(ground_truth, 1)
            metadata = {
                "reward_model": data['reward_model']
            }
            output = asyncio.run(reward_fn(
                None, 
                SimpleNamespace(prompt=test_prompt, response=test_response, metadata=metadata
            )))
            print(f"{output['reward_value']=}")
            # assert output["reward_value"] == expected_reward
            # break
