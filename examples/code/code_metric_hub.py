from collections import defaultdict

class CodeMetricGatherer:
    def __init__(self):
        self._dynamic_filter_drop_reason_count = defaultdict(lambda: 0)
        self.code_reward_time_lst = []
        self.code_execute_time_lst = []
        self.code_execute_time_max_lst = []
        self.code_reward_error_cnt = 0
        self.no_code_extract_cnt = 0

    def on_dynamic_filter_drop(self, reason: str | None):
        if not reason:
            return
        self._dynamic_filter_drop_reason_count[reason] += 1

    def log_code_execute_status(self, code_execute_status_lst: list[dict[str, int]]):
        for code_execute_status in code_execute_status_lst:
            self.code_reward_time_lst.append(code_execute_status.get("code_reward_time", None))
            self.code_execute_time_lst.append(code_execute_status.get("code_execute_time", None))
            self.code_reward_error_cnt += code_execute_status.get("code_reward_error", 0)
            self.no_code_extract_cnt += code_execute_status.get("no_code_extract", 0)
            self.code_execute_time_max_lst.append(code_execute_status.get("code_execute_time_max", None))
        
    def collect(self):
        metrics = {
            f"rollout/dynamic_filter/drop_{reason}": count
            for reason, count in self._dynamic_filter_drop_reason_count.items()
        }
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
        metrics['code/no_code_extract_cnt'] = self.no_code_extract_cnt
        return metrics