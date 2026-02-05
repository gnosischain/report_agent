"""
Cost tracking for OpenAI API usage.

Aggregates token usage across multiple API calls and provides cost estimates.
Thread-safe for use with parallel metric processing.

Usage:
    from report_agent.utils.cost_tracker import CostTracker, get_cost_tracker
    
    # Get the singleton tracker
    tracker = get_cost_tracker()
    
    # Record usage after API call
    tracker.record_usage(
        category="per_metric",
        model="gpt-4.1",
        input_tokens=1500,
        output_tokens=800,
        metric_name="api_p2p_discv4_clients_daily",
    )
    
    # Print summary at end of run
    tracker.print_summary()
    
    # Reset for next run
    tracker.reset()
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# Approximate pricing per 1K tokens (as of early 2025)
# These should be updated as OpenAI changes pricing
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4.1": {"input": 0.01, "output": 0.03},
    "gpt-4.1-mini": {"input": 0.0004, "output": 0.0016},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    # Default fallback
    "default": {"input": 0.01, "output": 0.03},
}


@dataclass
class UsageRecord:
    """Single API call usage record."""
    category: str  # "per_metric", "cross_metric", "summary"
    model: str
    input_tokens: int
    output_tokens: int
    metric_name: Optional[str] = None  # For per-metric calls
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    def estimated_cost(self) -> float:
        """Calculate estimated cost based on model pricing."""
        pricing = MODEL_PRICING.get(self.model, MODEL_PRICING["default"])
        input_cost = (self.input_tokens / 1000) * pricing["input"]
        output_cost = (self.output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class CostTracker:
    """
    Aggregates API usage across a run and provides cost summaries.
    
    Thread-safe for use with concurrent metric processing.
    """
    records: List[UsageRecord] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    
    def record_usage(
        self,
        category: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metric_name: Optional[str] = None,
    ) -> None:
        """
        Record usage from an API call.
        
        Args:
            category: Type of call ("per_metric", "cross_metric", "summary")
            model: Model name (e.g., "gpt-4.1")
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            metric_name: For per-metric calls, the metric being analyzed
        """
        record = UsageRecord(
            category=category,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metric_name=metric_name,
        )
        
        with self._lock:
            self.records.append(record)
        
        # Log individual call
        cost = record.estimated_cost()
        if metric_name:
            log.debug(
                f"API usage [{category}] {metric_name}: "
                f"{input_tokens:,} in + {output_tokens:,} out = {record.total_tokens:,} tokens (~${cost:.4f})"
            )
        else:
            log.debug(
                f"API usage [{category}]: "
                f"{input_tokens:,} in + {output_tokens:,} out = {record.total_tokens:,} tokens (~${cost:.4f})"
            )
    
    def get_summary(self) -> Dict[str, dict]:
        """
        Get aggregated summary by category.
        
        Returns:
            Dict with keys for each category containing:
            - calls: number of API calls
            - input_tokens: total input tokens
            - output_tokens: total output tokens
            - total_tokens: combined tokens
            - estimated_cost: estimated cost in USD
        """
        summary: Dict[str, dict] = {}
        
        with self._lock:
            for record in self.records:
                cat = record.category
                if cat not in summary:
                    summary[cat] = {
                        "calls": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "estimated_cost": 0.0,
                    }
                summary[cat]["calls"] += 1
                summary[cat]["input_tokens"] += record.input_tokens
                summary[cat]["output_tokens"] += record.output_tokens
                summary[cat]["total_tokens"] += record.total_tokens
                summary[cat]["estimated_cost"] += record.estimated_cost()
        
        return summary
    
    def get_total(self) -> dict:
        """Get total usage across all categories."""
        summary = self.get_summary()
        total = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
        }
        for cat_data in summary.values():
            total["calls"] += cat_data["calls"]
            total["input_tokens"] += cat_data["input_tokens"]
            total["output_tokens"] += cat_data["output_tokens"]
            total["total_tokens"] += cat_data["total_tokens"]
            total["estimated_cost"] += cat_data["estimated_cost"]
        return total
    
    def print_summary(self) -> None:
        """Print a formatted cost summary to stdout."""
        summary = self.get_summary()
        total = self.get_total()
        
        if total["calls"] == 0:
            print("\nNo API calls recorded.")
            return
        
        # Category display names
        cat_names = {
            "per_metric": "Per-metric analysis",
            "cross_metric": "Cross-metric analysis",
            "summary": "Weekly summary",
        }
        
        print("\n" + "=" * 60)
        print("API Cost Summary")
        print("=" * 60)
        
        for cat, data in summary.items():
            name = cat_names.get(cat, cat)
            print(
                f"  {name:.<30} {data['calls']:>3} calls, "
                f"{data['total_tokens']:>8,} tokens, ~${data['estimated_cost']:.2f}"
            )
        
        print("-" * 60)
        print(
            f"  {'Total':.<30} {total['calls']:>3} calls, "
            f"{total['total_tokens']:>8,} tokens, ~${total['estimated_cost']:.2f}"
        )
        print("=" * 60)
        print("  Note: Cost estimates are approximate based on public pricing.")
    
    def reset(self) -> None:
        """Clear all recorded usage (for starting a new run)."""
        with self._lock:
            self.records.clear()


# Singleton instance
_tracker: Optional[CostTracker] = None
_tracker_lock = threading.Lock()


def get_cost_tracker() -> CostTracker:
    """Get the singleton CostTracker instance."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = CostTracker()
    return _tracker


def reset_cost_tracker() -> None:
    """Reset the singleton tracker for a new run."""
    tracker = get_cost_tracker()
    tracker.reset()
