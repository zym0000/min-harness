from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from tool_execute_process.error_classify import ToolErrorType

@dataclass
class ToolMetrices:
    call_count: int = 0
    error_count: int = 0
    retry_count: int = 0
    success_count: int = 0
    total_latency_ms: float = 0.0
    error_type_distribution: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))

class MetricsControll:
    """监控埋点"""

    COST_PER_1K_TOKEN = {
        "deepseek": 0.00015,
    }

    def __init__(self):
        self.tool_metrics: Dict[str, ToolMetrices] = defaultdict(ToolMetrices)
        self.total_llm_calls: int = 0
        self.total_llm_tokens: int = 0
        self.total_tool_calls: int = 0
        self.total_retry_count: int = 0
        self.task_count: int = 0
        self.task_latency_ms: float = 0.0

    def record_tool_call(self, tool_name, latency_ms, success,
                         retry_count=0, error_type: Optional[ToolErrorType] = None):
        metric_tool = self.tool_metrics[tool_name]
        metric_tool.call_count += 1
        metric_tool.total_latency_ms += latency_ms
        metric_tool.retry_count += retry_count
        self.total_retry_count += retry_count

        if success:
            metric_tool.success_count += 1
        else:
            metric_tool.error_count += 1
            if error_type is not None:
                metric_tool.error_type_distribution[error_type.name] += 1

        self.total_tool_calls += 1

    def record_llm_call(self, token_used=0):
        self.total_llm_calls += 1
        self.total_llm_tokens += token_used

    def record_task(self, latency_ms):
        self.task_count += 1
        self.task_latency_ms += latency_ms

    def estimation_cost(self, model="deepseek"):
        cost_1k_token = self.COST_PER_1K_TOKEN.get(model, 0.00015)
        return (self.total_llm_tokens / 1000) * cost_1k_token

    def get_summary(self, model="deepseek") -> Dict[str, Any]:
        total_call = sum(m.call_count for m in self.tool_metrics.values())
        total_success = sum(m.success_count for m in self.tool_metrics.values())

        return {
            "total_llm_calls": self.total_llm_calls,
            "total_llm_tokens": self.total_llm_tokens,
            "tasks_completed": self.task_count,
            "total_retries": self.total_retry_count,
            "overall_success_rate": f"{total_success / total_call * 100:.1f}%" if total_call > 0 else "N/A",
            "avg_task_latency_ms": f"{self.task_latency_ms / self.task_count:.1f}" if self.task_count > 0 else "N/A",
            "estimation_cost": self.estimation_cost(model),
            "per_tool_summary": {
                name: {
                    "calls": m.call_count,
                    "retrys": m.retry_count,
                    "success_rate": f"{m.success_count / m.call_count * 100:.1f}%" if m.call_count > 0 else "N/A",
                    "avg_latency_ms": f"{m.total_latency_ms / m.call_count:.1f}" if m.call_count > 0 else "N/A",
                    "error_distribution": dict(m.error_type_distribution),
                }
                for name, m in self.tool_metrics.items()
            },
        }

    def print_summary(self):
        summary = self.get_summary()

        print(f"completed taks: {summary['tasks_completed']}")
        print(f"llm call count: {summary['total_llm_calls']}")
        print(f"total successed rate: {summary['overall_success_rate']}")
        print(f"total retry count: {summary['total_retries']}")
        print(f"average task latency time: {summary['avg_task_latency_ms']}")
        print(f"cost estimation: {summary['estimation_cost']}")

        for name, data in summary["per_tool_summary"].items():
            calls = data["calls"]
            rate = data["success_rate"]
            latency = data["avg_latency_ms"]
            retrys = data["retrys"]
            print(f"{name}: {calls} calls, success rate: {rate}, "
                  f"average latency time: {latency}, retry count: {retrys}")