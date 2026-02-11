from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from slime.utils.metric_utils import  dict_add_prefix

@dataclass
class CodeExtraInfo:
    code_reward_error: bool = False

    no_code_extract: bool = True
    code_reward_time: float | None = None
    code_execute_time: float | None = None
    code_execute_time_max: float | None = None
    pass_fail_list: list | None = None
    extracted_code: str | None = None
    pass_rate: float = 0.
    meta_data: dict | None = None

    format_reward: float | None = None
    
class CodeMetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self.code_reward_time_lst = []
        self.code_execute_time_lst = []
        self.code_execute_time_max_lst = []
        self.code_reward_error_cnt = 0
        self.no_code_extract_cnt = 0
        self.success_at_turn = []
        self.pass_rate_lst = []

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def log_code_execute_status(self, code_extra_info_list: list[CodeExtraInfo]):
        for code_extra_info in code_extra_info_list:
            self.code_reward_time_lst.append(code_extra_info.code_reward_time)
            self.code_execute_time_lst.append(code_extra_info.code_execute_time)
            self.code_execute_time_max_lst.append(code_extra_info.code_execute_time_max)
            self.code_reward_error_cnt += code_extra_info.code_reward_error
            self.no_code_extract_cnt += code_extra_info.no_code_extract
            self.pass_rate_lst.append(code_extra_info.pass_rate)
    
    def log_code_sample_train_metadata(self, samples):
        if isinstance(samples[0].reward, dict) and 'extra_info' in samples[0].reward:
            self.log_code_execute_status([sample.reward['extra_info'] for sample in samples])
        for sample in samples:
            self.success_at_turn.append(sample.train_metadata['_turn_idx'])

    def safe_mean(self, lst):
        lst = [i for i in lst if i is not None]
        return sum(lst) / len(lst) if len(lst) else 0.

    def safe_max(self, lst):
        lst = [i for i in lst if i is not None]
        # -1 or 0
        return max(lst) if len(lst) else 0.

    def collect(self, add_prefix=True):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        tot = len(self.code_reward_time_lst)
        if tot == 0:
            return metrics
        correct_lst = [1 if p==1.0 else 0 for p in self.pass_rate_lst]
        successful_turns = [turn for turn in self.success_at_turn if turn > 0]

        metrics['pass@1'] = self.safe_mean(correct_lst)
        metrics['pass_rate/mean'] = self.safe_mean(self.pass_rate_lst)
        metrics['reward_time/mean'] = self.safe_mean(self.code_reward_time_lst)
        metrics['reward_time/max'] = self.safe_max(self.code_reward_time_lst)
        metrics['execute_time/mean'] = self.safe_mean(self.code_execute_time_lst)
        metrics['execute_time/max'] = self.safe_max(self.code_execute_time_lst)
        metrics['execute_time_max/mean'] = self.safe_mean(self.code_execute_time_max_lst)
        metrics['execute_time_max/max'] = self.safe_max(self.code_execute_time_max_lst)
        metrics['reward_error_cnt'] = self.code_reward_error_cnt
        metrics['reward_error_cnt_ratio'] = self.code_reward_error_cnt / tot
        metrics['no_code_extract_cnt'] = self.no_code_extract_cnt
        metrics['no_code_extract_ratio'] = self.no_code_extract_cnt / tot
        metrics['success_at_turn/mean'] = self.safe_mean(successful_turns)
        if add_prefix:
            metrics = dict_add_prefix(metrics, "code/")
        return metrics