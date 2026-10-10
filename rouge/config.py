from __future__ import annotations

import json
from pathlib import Path

from .items import CARD_SLOTS, RELIC

COMPLEMENT_METHODS = ("complement", "complement_direct", "complement_random",
                      "complement_crossover", "complement_total")
METHODS = ("roguelike", "greedy", "random") + COMPLEMENT_METHODS

ACTIONS = ("direct", "forge", "complete", "fuse", "remove")
COMBO_ACTIONS = ("complete", "fuse")
POTIONS = ("scout", "duplicate", "d6")


COST_SCOPES = ("action", "round")
EVOLVABLE_SLOTS = CARD_SLOTS + (RELIC,)
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]

DEFAULTS = {
    "start_build": "starter",
    "proposal": {"evidence_chars": 6000, "repair_attempts": 1},
    "search": {"confirm_size": 4, "train_limit": 0, "max_relics": 3, "warmup_rounds": 6,
               "exploration": 1.0, "min_validation_gain": 0.0, "disabled_actions": [],
               "disabled_slots": [], "screen_size": 0, "screen_top_k": 1,
               "validation_chunk_size": 0, "pair_audit_probability": 1.0,
               "max_seconds": 0, "panel_reuse_rounds": 1, "screen_explore_probability": 1.0,
               "adaptive_controller": True},
    "chemistry": {"enabled": True, "min_tasks": 8, "prior_tasks": 4, "solo_threshold": 0.1,
                  "pair_threshold": 0.1},
    "potions": {"enabled": True, "capacity": 3, "start": ["scout"], "drop_probability": 0.3},
    "reward": {"regression_weight": 0.0, "cost_scope": "action"},
    "complement": {"consumer_slots": ["plan", "edit", "verify"],
                   "support_kinds": ["structure", "typed_values"], "weak_capacity": 4,
                   "supports_per_project": 1,
                   "weak_ttl_rounds": 3, "probe_projects": 2, "completion_projects": 1,
                   "q_attempts": 2, "explore_every": 4, "min_interaction": 0.0},
}


KNOWN = {
    "": ("seed", "method", "dataset_manifest", "start_build", "model", "budget", "execution", "docker",
         "proposal", "search", "chemistry", "potions", "reward", "complement"),
    "budget": ("development", "test", "tool_price"),
    "execution": ("max_steps", "max_tokens", "temperature", "top_p", "observation_chars", "parallel_tasks",
                  "max_error_rate", "reveal_expected", "progress_every", "preview_chars", "support_chars",
                  "training_trace_chars", "training_cell_credit"),
    "proposal": ("max_tokens", "diagnosis_tokens", "evidence_chars", "repair_attempts", "temperature"),
    "search": ("rounds", "candidates", "batch_size", "confirm_size", "train_limit", "validation_limit",
               "archive_size", "archive_probability", "warmup_rounds", "exploration", "min_validation_gain",
               "max_relics", "disabled_actions", "disabled_slots", "screen_size", "screen_top_k",
               "validation_chunk_size", "pair_audit_probability", "max_seconds", "panel_reuse_rounds",
               "screen_explore_probability", "adaptive_controller"),
    "chemistry": ("enabled", "min_tasks", "prior_tasks", "solo_threshold", "pair_threshold"),
    "potions": ("enabled", "capacity", "start", "drop_probability"),
    "reward": ("cost_weight", "regression_weight", "cost_scope"),
    "complement": ("consumer_slots", "support_kinds", "supports_per_project", "weak_capacity", "weak_ttl_rounds",
                   "probe_projects", "completion_projects", "q_attempts", "explore_every",
                   "min_interaction", "demand_aware", "probe_budget_fraction", "response_guided",
                   "refinement_chars", "post_warmup_candidates", "selection_candidates",
                   "max_project_attempts", "completion_budget_fraction", "frontier_guided",
                   "frontier_capacity", "frontier_quality_weight", "frontier_min_gain",
                   "frontier_max_regressions", "frontier_explore_probability",
                   "validation_every", "selection_budget_fraction", "combination_offers",
                   "expedition_guided", "producer_exploration_attempts",
                   "producer_exploration_budget_fraction", "selection_reserve_candidates",
                   "campaign_guided", "campaign_ablation", "campaign_candidates",
                   "campaign_expand_candidates", "campaign_composition_offers", "campaign_composition_pool",
                   "campaign_audit_budget_fraction", "campaign_route_exploration", "campaign_cost_safety",
                   "campaign_unknown_producer_attempts"),
}


def load_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, value in DEFAULTS.items():
        if isinstance(value, dict):
            config.setdefault(key, {})
            for sub, default in value.items():
                config[key].setdefault(sub, default)
        else:
            config.setdefault(key, value)
    return validate(config)


def resolve_path(value: str | Path, base: Path = EXPERIMENT_DIR) -> str:

    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (base / path).resolve())


def validate(config: dict) -> dict:
    if config.get("method") == "completion":
        raise ValueError('method "completion" was renamed to "roguelike"')
    if config.get("method") not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    for section, allowed in KNOWN.items():
        node = config if not section else config.get(section)
        if isinstance(node, dict):
            unknown = sorted(k for k in node if k not in allowed and not k.startswith("_"))
            if unknown:
                raise ValueError(f"Unknown config key(s) in {section or 'top level'}: {unknown}")
    _check(config, "seed", int, lambda x: x >= 0)
    _check(config, "dataset_manifest", str)
    _check(config, "start_build", str)
    _check(config, "model", dict)
    model = config["model"]
    backend = model.get("backend", "hf")
    if backend == "openai":
        _check(model, "base_url", str)
        if not model["base_url"].startswith(("http://", "https://")):
            raise ValueError("model.base_url must be http or https")
    elif backend != "hf":
        raise ValueError(f"Unknown model backend {backend!r}")
    _check(model, "path", str)
    _check(model, "context_tokens", int, lambda x: 512 <= x <= 1_000_000, missing_ok=True)
    _check(model, "enable_thinking", bool, missing_ok=True)
    _check(model, "batch_size", int, lambda x: 1 <= x <= 8, missing_ok=True)
    _check(model, "batch_prefill_tokens", int, lambda x: x >= 512, missing_ok=True)

    _check(config, "budget", dict)
    for key in ("development", "test", "tool_price"):
        _check(config["budget"], key, (int, float), lambda x: x > 0)

    execution = _section(config, "execution")
    for key in ("max_steps", "max_tokens", "observation_chars"):
        _check(execution, key, int, lambda x: x > 0)
    _check(execution, "temperature", (int, float), lambda x: 0 <= x <= 2)
    _check(execution, "top_p", (int, float), lambda x: 0 < x <= 1, missing_ok=True)
    _check(execution, "parallel_tasks", int, lambda x: x >= 1, missing_ok=True)
    _check(execution, "max_error_rate", (int, float), lambda x: 0 < x <= 1, missing_ok=True)
    _check(execution, "reveal_expected", bool, missing_ok=True)
    _check(execution, "preview_chars", int, lambda x: x >= 0, missing_ok=True)
    _check(execution, "support_chars", int, lambda x: 256 <= x <= 6000, missing_ok=True)
    _check(execution, 'training_trace_chars', int, lambda x: 512 <= x <= 16000, missing_ok=True)
    _check(execution, 'training_cell_credit', bool, missing_ok=True)

    docker = _section(config, "docker")
    _check(docker, "image", str)
    _check(docker, "timeout_seconds", int, lambda x: x > 0, missing_ok=True)

    proposal = _section(config, "proposal")
    for key in ("max_tokens", "diagnosis_tokens", "evidence_chars"):
        _check(proposal, key, int, lambda x: x > 0)
    _check(proposal, "repair_attempts", int, lambda x: 0 <= x <= 3)
    _check(proposal, "temperature", (int, float), lambda x: 0 <= x <= 2)

    search = _section(config, "search")
    for key in ("rounds", "candidates", "batch_size", "validation_limit", "archive_size", "warmup_rounds"):
        _check(search, key, int, lambda x: x > 0)
    for key in ("confirm_size", "train_limit", "max_relics"):
        _check(search, key, int, lambda x: x >= 0)
    _check(search, "archive_probability", (int, float), lambda x: 0 <= x <= 1)
    _check(search, "exploration", (int, float), lambda x: x >= 0)
    _check(search, "min_validation_gain", (int, float), lambda x: x >= 0)
    for key in ("screen_size", "validation_chunk_size", "max_seconds"):
        _check(search, key, int, lambda x: x >= 0)
    _check(search, "screen_top_k", int, lambda x: x >= 1)
    _check(search, "pair_audit_probability", (int, float), lambda x: 0 <= x <= 1)
    _check(search, "panel_reuse_rounds", int, lambda x: x >= 1)
    _check(search, "screen_explore_probability", (int, float), lambda x: 0 <= x <= 1)
    _check(search, "adaptive_controller", bool)
    if search["screen_size"] > search["batch_size"]:
        raise ValueError("screen_size must not exceed batch_size")
    _subset(search, "disabled_actions", ACTIONS)
    if "direct" in search["disabled_actions"]:
        raise ValueError("the direct action cannot be disabled")
    _subset(search, "disabled_slots", EVOLVABLE_SLOTS)
    if set(search["disabled_slots"]) >= set(EVOLVABLE_SLOTS):
        raise ValueError("at least one slot must stay evolvable")

    chemistry = _section(config, "chemistry")
    _check(chemistry, "enabled", bool)
    for key in ("min_tasks", "prior_tasks"):
        _check(chemistry, key, int, lambda x: x >= 1)
    for key in ("solo_threshold", "pair_threshold"):
        _check(chemistry, key, (int, float), lambda x: 0 <= x <= 1)

    potions = _section(config, "potions")
    _check(potions, "enabled", bool)
    _check(potions, "capacity", int, lambda x: x >= 0)
    _subset(potions, "start", POTIONS, unique=False)
    if len(potions["start"]) > potions["capacity"]:
        raise ValueError("potions.start exceeds the belt capacity")
    _check(potions, "drop_probability", (int, float), lambda x: 0 <= x <= 1)

    reward = _section(config, "reward")
    _check(reward, "cost_weight", (int, float), lambda x: x >= 0)
    _check(reward, "regression_weight", (int, float), lambda x: x >= 0)
    _check(reward, "cost_scope", str, lambda x: x in COST_SCOPES)
    complement = _section(config, "complement")
    _subset(complement, "consumer_slots", ("plan", "edit", "verify"))
    from .support import SUPPORT_KINDS
    _subset(complement, "support_kinds", SUPPORT_KINDS)
    if not complement["consumer_slots"] or not complement["support_kinds"]:
        raise ValueError("complement consumer_slots and support_kinds must be nonempty")
    for key in ("supports_per_project", "weak_capacity", "weak_ttl_rounds", "probe_projects", "completion_projects",
                "q_attempts", "explore_every"):
        _check(complement, key, int, lambda x: x >= 1)
    _check(complement, "min_interaction", (int, float), lambda x: x >= 0)
    _check(complement, "demand_aware", bool, missing_ok=True)
    _check(complement, "probe_budget_fraction", (int, float), lambda x: 0 <= x <= 1, missing_ok=True)
    _check(complement, 'response_guided', bool, missing_ok=True)
    _check(complement, 'frontier_guided', bool, missing_ok=True)
    _check(complement, 'expedition_guided', bool, missing_ok=True)
    _check(complement, 'campaign_guided', bool, missing_ok=True)
    if complement.get("demand_aware", False):
        from .demands import FACT_KINDS
        if config['method'] not in ('complement', 'complement_direct'):
            raise ValueError('Demand-aware v3 currently supports complement and complement_direct')
        if not set(complement["support_kinds"]) <= set(FACT_KINDS):
            raise ValueError("Demand-aware search requires declared fact-interface support kinds")
        if "probe_budget_fraction" not in complement:
            raise ValueError("Demand-aware search requires an explicit probe budget fraction")
    if complement.get('response_guided', False):
        if not complement.get('demand_aware', False):
            raise ValueError('Response-guided v4 requires demand_aware=true')
        for key in ('post_warmup_candidates', 'selection_candidates', 'max_project_attempts'):
            _check(complement, key, int, lambda x: 1 <= x <= 4)
        maximum = 2500 if complement.get('frontier_guided', False) else 1500
        _check(complement, 'refinement_chars', int, lambda x: 200 <= x <= maximum)
        _check(complement, 'completion_budget_fraction', (int, float), lambda x: 0 <= x <= 1)
    if complement.get('frontier_guided', False):
        if not complement.get('response_guided', False):
            raise ValueError('Frontier-guided v5 requires response_guided=true')
        _check(complement, 'frontier_capacity', int, lambda x: 2 <= x <= 8)
        _check(complement, 'frontier_quality_weight', (int, float), lambda x: 0 < x <= 1)
        _check(complement, 'frontier_min_gain', (int, float), lambda x: 0 < x <= 1)
        _check(complement, 'frontier_max_regressions', int, lambda x: 0 <= x <= 2)
        _check(complement, 'frontier_explore_probability', (int, float), lambda x: 0 <= x <= 1)
        _check(complement, 'validation_every', int, lambda x: 1 <= x <= 10)
        _check(complement, 'selection_budget_fraction', (int, float), lambda x: 0.1 <= x <= 0.5)
        _check(complement, 'combination_offers', int, lambda x: 1 <= x <= 4)
        if search['confirm_size'] < 2:
            raise ValueError('Frontier-guided search requires at least two confirmation tasks')
    if complement.get('expedition_guided', False):
        if not complement.get('frontier_guided', False):
            raise ValueError('Expedition v6 requires frontier_guided=true')
        _check(complement, 'producer_exploration_attempts', int, lambda x: 0 <= x <= 4)
        _check(complement, 'producer_exploration_budget_fraction', (int, float), lambda x: 0 <= x <= 0.3)
        _check(complement, 'selection_reserve_candidates', int, lambda x: 1 <= x <= 2)
        if complement['producer_exploration_budget_fraction'] > complement['completion_budget_fraction']:
            raise ValueError('Producer exploration must fit inside the completion allocation')
        if search['min_validation_gain'] != 0:
            raise ValueError('Expedition paired tie selection requires min_validation_gain=0')
    if complement.get('campaign_guided', False):
        if not complement.get('expedition_guided', False) or config['method'] != 'complement':
            raise ValueError('Campaign v7 requires expedition_guided=true and method=complement')
        _check(complement, 'campaign_ablation', str, lambda x: x in
               ('full', 'no_composition', 'no_potion', 'uniform_route',
                'no_ordinary_joint', 'within_round_partners'))
        _check(complement, 'campaign_candidates', int, lambda x: 2 <= x <= 4)
        _check(complement, 'campaign_expand_candidates', int, lambda x: 1 <= x <= 2)
        _check(complement, 'campaign_composition_offers', int, lambda x: 1 <= x <= 2)
        _check(complement, 'campaign_composition_pool', int, lambda x: 2 <= x <= 16)
        _check(complement, 'campaign_audit_budget_fraction', (int, float), lambda x: 0 <= x <= 0.15)
        _check(complement, 'campaign_route_exploration', (int, float), lambda x: 0.1 <= x <= 1)
        _check(complement, 'campaign_cost_safety', (int, float), lambda x: 1.1 <= x <= 2)
        _check(complement, 'campaign_unknown_producer_attempts', int, lambda x: 0 <= x <= 1)
    if complement["supports_per_project"] > len(complement["support_kinds"]):
        raise ValueError("supports_per_project exceeds available support kinds")
    if config["method"] in COMPLEMENT_METHODS:
        if search["warmup_rounds"] < 2:
            raise ValueError("complement search requires at least two warmup rounds")
        if search["batch_size"] <= (search["screen_size"] or search["batch_size"]):
            raise ValueError("complement search requires a screen smaller than the training batch")
        if search["confirm_size"] < 1:
            raise ValueError("complement search requires disjoint confirmation tasks")
        if not set(complement["consumer_slots"]) - set(search["disabled_slots"]):
            raise ValueError("No complement consumer slot is enabled")
        if "inspect" in search["disabled_slots"]:
            raise ValueError("complement requires inspect for permanent input producers")
    return config


def _section(config: dict, key: str) -> dict:
    _check(config, key, dict)
    return config[key]


def _subset(config: dict, key: str, allowed: tuple, unique: bool = True) -> None:
    _check(config, key, list)
    values = config[key]
    if not all(isinstance(x, str) for x in values) or not set(values) <= set(allowed):
        raise ValueError(f"{key} must only contain {allowed}")
    if unique and len(set(values)) != len(values):
        raise ValueError(f"{key} has duplicates")


def _check(config: dict, key: str, typ, pred=None, missing_ok=False):
    if key not in config:
        if missing_ok:
            return
        raise ValueError(f"Missing config key: {key}")
    value = config[key]
    if isinstance(value, bool) and typ in (int, float, (int, float)):
        raise ValueError(f"{key} must be a number, not a boolean")
    if not isinstance(value, typ):
        raise ValueError(f"{key} must be {typ}")
    if pred and not pred(value):
        raise ValueError(f"{key}={value!r} fails validation")
