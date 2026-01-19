from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
import random
from typing import Any, Dict, List, Optional


class EvalCache:
    """LRU cache: signature(str) -> eval(dict)"""

    def __init__(self, max_size: int = 5000):
        self.max_size = int(max_size)
        self._d: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self._d:
            v = self._d.pop(key)
            self._d[key] = v
            self.hit_count += 1
            return v
        self.miss_count += 1
        return None

    def put(self, key: str, val: Dict[str, Any]) -> None:
        if key in self._d:
            self._d.pop(key)
        self._d[key] = val
        while len(self._d) > self.max_size:
            self._d.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        if total <= 0:
            return 0.0
        return float(self.hit_count) / float(total)


@dataclass
class BanditArm:
    name: str
    value: float = 0.0
    count: int = 0


class EpsGreedyBandit:
    def __init__(self, arms: List[str], eps: float = 0.1, seed: int = 0):
        self.eps = float(eps)
        self.rng = random.Random(int(seed))
        self.arms = {a: BanditArm(name=a) for a in arms}

    def choose(self) -> str:
        if self.rng.random() < self.eps:
            return self.rng.choice(list(self.arms.keys()))
        best = max(self.arms.values(), key=lambda x: x.value)
        return best.name

    def update(self, arm: str, reward: float) -> None:
        a = self.arms[arm]
        a.count += 1
        a.value += (float(reward) - a.value) / max(1, a.count)


class PolicySwitchController:
    """
    v5.4 minimal policy-switch + bandit:
    - action_family bandit: choose which move family to sample
    - policy bandit: choose "heuristic" vs "llm" (if enabled)
    """

    def __init__(
        self,
        action_families: List[str],
        policies: List[str],
        eps: float,
        seed: int,
    ):
        self.action_bandit = EpsGreedyBandit(action_families, eps=eps, seed=seed)
        self.policy_bandit = EpsGreedyBandit(policies, eps=eps, seed=seed + 17)
        self.last_action_family: Optional[str] = None
        self.last_policy: Optional[str] = None

    def choose_action_family(self) -> str:
        self.last_action_family = self.action_bandit.choose()
        return self.last_action_family

    def choose_policy(self) -> str:
        self.last_policy = self.policy_bandit.choose()
        return self.last_policy

    def update(self, improved: bool, delta_total: float) -> None:
        """
        reward: improvement => positive reward; else small negative
        """
        r = 1.0 if improved else -0.1
        r += max(-1.0, min(1.0, -float(delta_total)))
        if self.last_action_family:
            self.action_bandit.update(self.last_action_family, r)
        if self.last_policy:
            self.policy_bandit.update(self.last_policy, r)
