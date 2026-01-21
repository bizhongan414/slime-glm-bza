import asyncio
import logging
import re
import json
import time
from types import SimpleNamespace

from .sandbox_utils import execute_code
from .code_metric import CodeExtraInfo

logger = logging.getLogger(__name__)

_TIMEOUT = 60

class RewardFn:
    def __init__(self, args):
        self.default_time_limit_s = args.sandbox_default_time_limit_s
        self.default_memory_limit_mb = args.sandbox_default_memory_limit_mb
        self.local_run = args.sandbox_local_run

        self.sandbox_fusion_url = args.sandbox_url if not self.local_run else None
        self.use_case_custom_time_limit = args.sandbox_use_case_custom_time_limit
        self.use_case_custom_memory_limit = args.sandbox_use_case_custom_memory_limit
        
        self.max_turns = args.code_rollout_max_turn
        self.use_format_reward = args.code_use_format_reward
        self.use_timeout_reward = args.code_use_timeout_reward

    async def __call__(self, sample, **kwargs):
        start_time = time.monotonic()
        # TODO for debug, save all, future only keep necessary keys
        extra_info = CodeExtraInfo()
        meta_data = None
        pass_rate = None
        score = 0
        try:
            answer_reward = 0.0
            format_reward = 0.0
            timeout_reward = 0.0

            extracted_code = _extract_code_from_answer(sample.response)
            if extracted_code is not None:
                extra_info.no_code_extract = False
                extra_info.extracted_code = extracted_code
                
                reward_model = sample.metadata['reward_model']
                ground_truth = json.loads(reward_model['ground_truth'])
                
                time_limit = self.use_case_custom_time_limit and reward_model.get("time_limit", None) or self.default_time_limit_s
                memory_limit_mb = self.use_case_custom_memory_limit and reward_model.get("memory_limit_mb", None) or self.default_memory_limit_mb
                if self.local_run:
                    time_limit += 5

                response, exec_statu, meta_data = execute_code(
                    sandbox_fusion_url=self.sandbox_fusion_url,
                    memory_limit_mb=memory_limit_mb,
                    ground_truth=ground_truth,
                    code=extracted_code,
                    timeout=time_limit,
                    language="python",
                    local_run=self.local_run
                )
                extra_info.meta_data = meta_data

                duration_lst = meta_data.get("duration", None)
                if duration_lst is not None:
                    if not isinstance(duration_lst, list):
                        duration_lst = [duration_lst]
                    extra_info.code_execute_time = sum(duration_lst) / len(duration_lst)
                    extra_info.code_execute_time_max = max(duration_lst)
                    
                if exec_statu.lower() == "timeout":
                    logger.warning("execute code timeout, not implement")
                    reward_msg = "exec_timeout"
                else:
                    match_test_pass_rate = re.search(r"pass rate: \*\*(.*?)\*\*", meta_data["stdout"])
                    pass_rate = float(match_test_pass_rate.group(1)) if match_test_pass_rate else 1.0
                    logger.info(f"{pass_rate=}")
                    if pass_rate == 1.0:
                        score = 1
                        answer_reward = 1
                        reward_msg = "accept"
                    else:
                        answer_reward = 0
                        reward_msg = "wrong_answer"
            else:
                reward_msg = "no_code"
                extra_info.no_code_extract = True
            # reward = answer_reward + format_reward + timeout_reward
            reward = answer_reward
            
            extra_info.code_reward_time = round(time.monotonic() - start_time, 4)
            extra_info.pass_rate = pass_rate
            
            return dict(
                reward_value=reward,
                score=score,
                reward_cat=reward_msg,
                extra_info=extra_info,
            )
        except Exception as e:
            logger.warning(f"Error in RewardFn: {e=} {sample.prompt=} {sample.response=}")
            extra_info.code_reward_error = True
            extra_info.code_reward_time = round(time.monotonic() - start_time, 4)
            return dict(
                reward_value=0.0, 
                score=score,
                reward_cat="python_error",
                extra_info=extra_info,
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
