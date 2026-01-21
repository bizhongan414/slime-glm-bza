from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CodeExtraInfo:
    code_reward_error: bool = False

    no_code_extract: bool = True
    code_reward_time: float | None = None
    code_execute_time: float | None = None
    code_execute_time_max: float | None = None

    extracted_code: str | None = None
    pass_rate: float | None = None
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

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def log_code_execute_status(self, code_extra_info_list: list[CodeExtraInfo]):
        for code_extra_info in code_extra_info_list:
            self.code_reward_time_lst.append(code_extra_info.code_reward_time)
            if code_extra_info.code_execute_time:
                self.code_execute_time_lst.append(code_extra_info.code_execute_time)
                self.code_execute_time_max_lst.append(code_extra_info.code_execute_time_max)
            self.code_reward_error_cnt += code_extra_info.code_reward_error
            self.no_code_extract_cnt += code_extra_info.no_code_extract
            
            # self.code_reward_time_lst.append(code_execute_status.get("code_reward_time", None))
            # self.code_execute_time_lst.append(code_execute_status.get("code_execute_time", None))
            # self.code_reward_error_cnt += code_execute_status.get("code_reward_error", 0)
            # self.no_code_extract_cnt += code_execute_status.get("no_code_extract", 0)
            # self.code_execute_time_max_lst.append(code_execute_status.get("code_execute_time_max", None))
    
    def log_code_sample_train_metadata(self, samples):
        self.log_code_execute_status([sample.reward['extra_info'] for sample in samples])
        for sample in samples:
            self.success_at_turn.append(sample.train_metadata['_turn_idx'])

    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
        tot = len(self.code_reward_time_lst)
        self.code_reward_time_lst = [t for t in self.code_reward_time_lst if t is not None]
        self.code_execute_time_lst = [t for t in self.code_execute_time_lst if t is not None]
        self.code_execute_time_max_lst = [t for t in self.code_execute_time_max_lst if t is not None]
        metrics['code/code_reward_time/mean'] = sum(self.code_reward_time_lst) / len(self.code_reward_time_lst) if len(self.code_reward_time_lst) else 0
        metrics['code/code_reward_time/max'] = max(self.code_reward_time_lst) if len(self.code_reward_time_lst) else 0
        metrics['code/code_execute_time/mean'] = sum(self.code_execute_time_lst) / len(self.code_execute_time_lst) if len(self.code_execute_time_lst) else 0
        metrics['code/code_execute_time/max'] = max(self.code_execute_time_lst) if len(self.code_execute_time_lst) else 0
        metrics['code/code_execute_time_max/mean'] = sum(self.code_execute_time_max_lst) / len(self.code_execute_time_max_lst) if len(self.code_execute_time_max_lst) else 0
        metrics['code/code_execute_time_max/max'] = max(self.code_execute_time_max_lst) if len(self.code_execute_time_max_lst) else 0
        metrics['code/code_reward_error_cnt'] = self.code_reward_error_cnt
        metrics['code/code_reward_error_cnt_ratio'] = self.code_reward_error_cnt / tot
        metrics['code/no_code_extract_cnt'] = self.no_code_extract_cnt
        metrics['code/no_code_extract_ratio'] = self.no_code_extract_cnt / tot
        successful_turns = [turn for turn in self.success_at_turn if turn > 0]
        metrics['code/success_at_turn/mean'] = sum(successful_turns) / len(successful_turns) if len(successful_turns) > 0 else 0.0
        return metrics