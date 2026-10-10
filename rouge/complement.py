from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .core import (BudgetExceeded, Evaluation, Events, interaction, paired_delta,
                   rng_for, seed_for, write_json)
from .evolution import Context, evidence
from .items import Build, Item
from .racing import strict_gate
from .support import PRODUCER_SCHEMA, make_source_item, retain_inspection


@dataclass
class WeakItem:
    item: Item
    born: int


@dataclass
class Choice:
    build: Build
    evaluation: Evaluation
    tag: str
    p: Item | None = None
    q: Item | None = None


class ComplementSearch:
    def __init__(self, config, evaluator, proposer, events: Events, output: Path, start: Build):
        self.config, self.evaluator, self.proposer = config, evaluator, proposer
        self.events, self.output, self.budget = events, output, evaluator.budget
        self.seed, self.method = int(config["seed"]), config["method"]
        self.search, self.options = config["search"], config["complement"]
        self.start, self.best, self.best_validation = start, start, None
        self.slots = [s for s in self.options["consumer_slots"] if s not in self.search["disabled_slots"]]
        self.archive: dict[str, WeakItem] = {}
        self.seen: set[str] = {x.id for x in start.items}
        self.history, self.probes, self.permanent_pairs = [], [], []
        self.tasks = evaluator.dataset.tasks

    def save(self, build):
        write_json(self.output / "builds" / f"{build.id}.json", build.to_dict())

    def _context(self, build, evaluation, round_id):
        return Context(build, evidence(self.tasks, evaluation, rng_for(self.seed, "evidence", round_id),
                                       int(self.config["proposal"]["evidence_chars"])),
                       chemistry="Input-only support interface available to ALL controls:\n" +
                       PRODUCER_SCHEMA + "\nConsume useful observations when present; if absent, "
                       "inspect the inputs yourself. Do not require a fixed task-specific fact.")

    def _batches(self, train, round_id):
        order = rng_for(self.seed, "complement_batch", round_id).sample(train, len(train))
        size = int(self.search["batch_size"])
        panel = order[:size]
        confirm = order[size:size + int(self.search["confirm_size"])]
        if len(panel) != size or len(confirm) != int(self.search["confirm_size"]):
            raise ValueError("Not enough training tasks for disjoint panel and confirmation")
        return panel, confirm

    def _changes(self, build):
        return sum(self.best.get(x.key) != x for x in build.execution_items)

    def _order(self, choices, round_id):
        unique = {c.build.execution_id: c for c in choices
                  if c.build.execution_id != self.best.execution_id}
        rows = list(unique.values())
        rng_for(self.seed, "complement_ties", round_id).shuffle(rows)

        rows.sort(key=lambda c: (c.evaluation.score, -self._changes(c.build)), reverse=True)
        return rows

    def _normal_pair(self, p, q, base, tasks, round_id, tag):
        if p.key == q.key:
            raise ValueError("Consumer and producer must occupy distinct slots")
        q = self._prepare_q(q)
        hp, hq, hpq = self.best.with_item(p), self.best.with_item(q), self.best.compose(p, q)

        ep, eq, epq = [self.evaluator.evaluate(b, tasks) for b in (hp, hq, hpq)]
        row = {"round": round_id, "p": p.id, "q": q.id, "tag": tag,
               "task_ids": base.task_ids, "temporary_support": False,
               "scores": {"H": base.score, "Hp": ep.score, "Hq": eq.score, "Hpq": epq.score},
               "interaction": interaction(base, ep, eq, epq), "n": len(tasks),
               "evidence_status": "screen_only" if len(tasks) <= self.search["screen_size"] else
                                  "training_panel", "q_alone_at_least_pair": eq.score >= epq.score}
        self.permanent_pairs.append(row)
        self.events.emit("permanent_pair", **row)
        for b in (hp, hq, hpq):
            self.save(b)
        return [Choice(hp, ep, tag + ":p", p), Choice(hq, eq, tag + ":q", None, q),
                Choice(hpq, epq, tag + ":pq", p, q)]

    def _prepare_q(self, q):

        return retain_inspection(q, self.best)

    def _paired_support_evidence(self, p, kind, panel, full):

        hp = self.best.with_item(p)
        plain = self.evaluator.evaluate(hp, panel).by_task()
        assisted = self.evaluator.evaluate_supported(hp, panel, kind).by_task()
        parent = self.evaluator.evaluate(self.best, panel).by_task()
        priority = list(full["rescued"]) + list(full["regressed"]) + [
            t.id for t in panel if parent[t.id].success] + [t.id for t in panel]
        ids = list(dict.fromkeys(priority))[:3]
        limit = int(self.config["proposal"]["evidence_chars"])
        summary = (f"Support type: {kind}; H/Hp/Hz/Hpz scores: {full['scores']}; "
                   f"specific response: {full['interaction']:.3f}; "
                   f"assisted net gain: {full['assisted_gain']:.3f}. "
                   "These are investment probes, not deployment rewards.\n")
        each = max(1, (limit - len(summary)) // max(1, len(ids)))
        blocks = []
        for index, task_id in enumerate(ids, 1):
            task = self.tasks[task_id]
            facts = self.evaluator.support_evidence(hp, [task], kind, limit_chars=max(200, each // 3))
            block = (f"Aligned training example {index}: {task.instruction[:max(80, each // 5)]}\n"
                     f"Normal Hp success={plain[task_id].success}; actual trace tail:\n"
                     f"{plain[task_id].trace[-max(80, each // 5):]}\n"
                     f"Supported Hpz success={assisted[task_id].success}; actual trace tail:\n"
                     f"{assisted[task_id].trace[-max(80, each // 5):]}\n"
                     f"Input-only support facts for THIS example:\n{facts}")
            blocks.append(block[:each])
        return (summary + "\n\n".join(blocks))[:limit]

    def _probe(self, p, kind, base, panel, round_id, stage):
        before = self.budget.spent
        hp = self.best.with_item(p)
        ep = self.evaluator.evaluate(hp, panel)
        ez = self.evaluator.evaluate_supported(self.best, panel, kind)
        epz = self.evaluator.evaluate_supported(hp, panel, kind)
        change = paired_delta(base, epz)
        response = interaction(base, ep, ez, epz)
        row = {"round": round_id, "p": p.id, "support_kind": kind, "stage": stage,
               "task_ids": base.task_ids, "scores": {"H": base.score, "Hp": ep.score,
                                                       "Hz": ez.score, "Hpz": epz.score},
               "interaction": response, "assisted_gain": change.gain,
               "rescued": list(change.fixed), "regressed": list(change.regressed),
               "spent": self.budget.spent - before,
               "estimated_regeneration_cost": self.budget.tool_price * len(panel),
               "priority": response / max(1.0, self.budget.tool_price * len(panel)),
               "evidence_status": "screen_only" if stage == "screen" else "expanded_training",
               "eligible": change.net > 0 and response > float(self.options["min_interaction"])}
        self.probes.append(row)
        self.events.emit("support_probe", **row)
        return row

    def _projects(self, pool, base, screen, panel, round_id):

        rng = rng_for(self.seed, "complement_offer", round_id)
        pool = list(pool)
        rng.shuffle(pool)
        pool = pool[:int(self.options["probe_projects"])]
        if not pool:
            return []
        if self.method in ("complement_direct", "complement_crossover"):
            return [(p, rng.choice(self.options["support_kinds"]), "direct", "")
                    for p in pool[:int(self.options["completion_projects"])]]
        rows = []
        for p in pool:
            kinds = list(self.options["support_kinds"])
            rng.shuffle(kinds)
            for kind in kinds[:int(self.options["supports_per_project"])]:
                rows.append((p, self._probe(p, kind, base, screen, round_id, "screen")))
        exploratory = (round_id - int(self.search["warmup_rounds"]) + 1) % int(
            self.options["explore_every"]) == 0
        if self.method == "complement_random":
            rng.shuffle(rows)
            ranked, reason = rows, "random"
        elif exploratory:
            rng.shuffle(rows)
            ranked, reason = rows, "fixed_exploration_quota"
        elif self.method == "complement_total":
            ranked = sorted([(p, r) for p, r in rows if r["assisted_gain"] > 0],
                            key=lambda pr: pr[1]["scores"]["Hpz"], reverse=True)
            reason = "assisted_total"
        else:
            ranked = sorted([(p, r) for p, r in rows if r["eligible"]],
                            key=lambda pr: (pr[1]["priority"], pr[1]["assisted_gain"]), reverse=True)
            reason = "specific_response"


        if not ranked and round_id == int(self.search["warmup_rounds"]):
            rng.shuffle(rows)
            ranked, reason = rows, "initial_unknown_exploration"
        picked, used = [], set()
        for p, row in ranked:
            if p.id in used:
                continue
            used.add(p.id)

            full_base = self.evaluator.evaluate(self.best, panel)
            full = self._probe(p, row["support_kind"], full_base, panel, round_id, "expanded")
            exploit = reason in ("specific_response", "assisted_total")
            qualifies = full["eligible"] if self.method == "complement" else full["assisted_gain"] > 0
            if exploit and not qualifies:
                self.events.emit("support_project_rejected", round=round_id, p=p.id,
                                 support_kind=row["support_kind"], reason="response_did_not_survive_expansion")
                continue
            description = self._paired_support_evidence(p, row["support_kind"], panel, full)
            picked.append((p, row["support_kind"], reason, description))
            if len(picked) >= int(self.options["completion_projects"]):
                break
        return picked

    def _weak_pool(self, base, screen, round_id):
        weak = []
        for entry in self.archive.values():
            ep = self.evaluator.evaluate(self.best.with_item(entry.item), screen)
            if ep.score <= base.score:
                weak.append(entry.item)
        return weak

    def _record_probe_cost(self, amount):
        return None

    def _remember_parent(self, evaluation):
        return None

    def _remember_consumer(self, p, evaluation):
        return None

    def _generation_context(self, build, evaluation, panel, round_id):
        return self._context(build, evaluation, round_id)

    def _draft_consumer(self, slot, context, seed):
        return self.proposer.draft("fresh", slot, context, seed)

    def _consumer_count(self, warmup):
        return int(self.search['candidates'])

    def _consumer_slot(self, round_id, index, warmup):
        return (self.slots[(round_id + index) % len(self.slots)] if warmup else
                rng_for(self.seed, 'consumer_slot', round_id, index).choice(self.slots))

    def _can_start_round(self, round_id):
        return True

    def _complete_project(self, p, kind, support, base, screen, panel, round_id):
        choices = []

        try:
            choices += self._normal_pair(p, make_source_item(kind), base, screen, round_id,
                                         "source_adoption")
        except ValueError as exc:
            self.events.emit("invalid_source_adoption", round=round_id, p=p.id,
                             support_kind=kind, error=str(exc)[:500])
        hp = self.best.with_item(p)
        normal = self.evaluator.evaluate(hp, panel)
        editor_context = self._context(self.best, normal, round_id)
        if self.method == "complement_crossover":
            crossover = self.best.compose(p, self._prepare_q(make_source_item(kind)))
            editor_context = self._context(crossover, self.evaluator.evaluate(crossover, panel), round_id)
        for attempt in range(int(self.options["q_attempts"])):
            q = self.proposer.draft_producer(editor_context, p,
                seed_for(self.seed, "producer", round_id, p.id, attempt), support_evidence=support)
            if q is None:
                continue
            try:
                choices += self._normal_pair(p, q, base, screen, round_id, "synthesized")
            except ValueError as exc:
                self.events.emit("invalid_permanent_completion", round=round_id, p=p.id,
                                 q=q.id, error=str(exc)[:500])
        return choices

    def _selection_candidates(self):
        return 1

    def _promising(self, evaluation, parent):
        return evaluation.score > parent.score

    def floor(self, round_id, train, validation):
        start, incumbent = self.budget.spent, self.best
        panel, confirm = self._batches(train, round_id)
        screen = panel[:int(self.search["screen_size"])]
        warmup = round_id < int(self.search["warmup_rounds"])
        stages, accepted, verdict = {}, False, "no_gain"
        previous_validation = self.best_validation
        scale = max(1.0, self.evaluator.task_cost * len(panel))
        before = self.budget.spent
        base = self.evaluator.evaluate(self.best, screen)
        context = self._generation_context(self.best, base, panel, round_id)
        stages["parent_screen"] = self.budget.spent - before
        before = self.budget.spent
        choices = []

        for index in range(self._consumer_count(warmup)):
            slot = self._consumer_slot(round_id, index, warmup)
            p = self._draft_consumer(slot, context, seed_for(self.seed, "consumer", round_id, index))
            if p is None:
                continue
            if p.id in self.seen:
                self.events.emit("duplicate_consumer", round=round_id, p=p.id)
                continue
            if p.slot not in self.slots:
                raise ValueError("Consumer proposer returned an unsupported slot")
            self.seen.add(p.id)
            hp = self.best.with_item(p)
            ep = self.evaluator.evaluate(hp, screen)
            self._remember_consumer(p, ep)
            choices.append(Choice(hp, ep, "direct:p", p))
            self.archive[p.id] = WeakItem(p, round_id)
            self.save(hp)
        stages["consumer_generation_and_screen"] = self.budget.spent - before

        expired = [key for key, entry in self.archive.items()
                   if round_id - entry.born >= int(self.options["weak_ttl_rounds"]) or self.best.has(entry.item)]
        for key in expired:
            del self.archive[key]
            self.events.emit("weak_item_expired", round=round_id, p=key)
        while len(self.archive) > int(self.options["weak_capacity"]):
            key = min(self.archive, key=lambda k: (self.archive[k].born, k))
            del self.archive[key]
            self.events.emit("weak_archive_evicted", round=round_id, p=key)
        before = self.budget.spent
        try:
            weak = self._weak_pool(base, screen, round_id)
            projects = [] if warmup else self._projects(weak, base, screen, panel, round_id)
        finally:

            self._record_probe_cost(self.budget.spent - before)
        stages["weak_screen_and_support_probes"] = self.budget.spent - before
        before = self.budget.spent
        for p, kind, reason, support in projects:
            self.events.emit("completion_investment", round=round_id, p=p.id, support_kind=kind,
                             reason=reason, uses_intervention=bool(support))
            choices += self._complete_project(p, kind, support, base, screen, panel, round_id)
        stages["source_adoption_and_synthesis"] = self.budget.spent - before
        before = self.budget.spent
        ranked = self._order(choices, round_id)

        selected = ranked if warmup else ranked[:int(self.search["screen_top_k"])]
        full_base = self.evaluator.evaluate(self.best, panel)
        self._remember_parent(full_base)
        expanded = [Choice(c.build, self.evaluator.evaluate(c.build, panel), c.tag, c.p, c.q)
                    for c in selected]

        for c in list(expanded):
            if c.p is not None and c.q is not None:
                expanded += self._normal_pair(c.p, c.q, full_base, panel, round_id, "expanded_audit")
        expanded = self._order(expanded, round_id)
        winner = expanded[0] if expanded else None
        stages["normal_panel_and_removals"] = self.budget.spent - before
        before = self.budget.spent
        confirm_change, gate, checked = None, None, None
        selection_attempts = []
        for candidate in expanded[:self._selection_candidates()]:
            if not self._promising(candidate.evaluation, full_base):
                continue
            winner = candidate
            gate, checked = None, None
            a = self.evaluator.evaluate(self.best, confirm)
            b = self.evaluator.evaluate(winner.build, confirm)
            confirm_change = paired_delta(a, b)
            verdict = "confirmation_regression"
            if confirm_change.net >= 0:
                checked, gate = strict_gate(self.evaluator, winner.build, validation, self.best_validation,
                    float(self.search["min_validation_gain"]), int(self.search["validation_chunk_size"]),
                    self.events)
                verdict = "validation_no_strict_gain"
                if checked and checked.score > self.best_validation.score + float(self.search["min_validation_gain"]):
                    self.best, self.best_validation = winner.build, checked
                    accepted, verdict = True, "accepted"
            selection_attempts.append({"build": winner.build.id, "tag": winner.tag,
                                       "confirm_net": confirm_change.net, "gate": gate,
                                       "selection_score": checked.score if checked else None,
                                       "verdict": verdict})
            self.events.emit("selection_attempt", round=round_id, **selection_attempts[-1])
            if accepted:
                break
        stages["confirmation_and_validation"] = self.budget.spent - before
        total = self.budget.spent - start
        deployed_gain = self.best_validation.score - previous_validation.score if accepted else 0.0
        regression_rate = (len(confirm_change.regressed) / confirm_change.n) if confirm_change else 0.0
        penalty = float(self.config["reward"]["cost_weight"]) * total / scale
        row = {"round": round_id, "warmup": warmup, "accepted": accepted, "verdict": verdict,
               "parent": incumbent.id, "harness": self.best.id, "screen_ids": [t.id for t in screen],
               "panel_ids": [t.id for t in panel], "confirm_ids": [t.id for t in confirm],
               "projects": [{"p": p.id, "kind": k, "selection": r} for p, k, r, _ in projects],
               "candidates": [{"build": c.build.id, "tag": c.tag, "score": c.evaluation.score}
                              for c in expanded],
               "best_candidate": winner.build.id if winner else None,
               "confirm_net": confirm_change.net if confirm_change else None,
               "validation_gate": gate, "validation_score": self.best_validation.score,
               "selection_attempts": selection_attempts,
               "reward_source": "accepted_deployment_validation", "deployed_gain": deployed_gain,
               "regression_rate": regression_rate, "cost_units": total / scale,
               "cost_penalty": penalty, "reward": deployed_gain - penalty -
                   float(self.config["reward"]["regression_weight"]) * regression_rate,
               "round_spent": total, "spent": self.budget.spent, "stage_gold": stages,
               "archive_size": len(self.archive)}
        self.history.append(row)
        self.events.emit("complement_round_complete", **row)
        self.persist()
        print(f"round={round_id} method={self.method} warmup={warmup} projects={len(projects)} "
              f"verdict={verdict} validation={self.best_validation.score:.3f} "
              f"spent={self.budget.spent:.0f}/{self.budget.limit:.0f}", flush=True)

    def persist(self):
        self.save(self.best)
        write_json(self.output / "best_build.json", self.best.to_dict())
        write_json(self.output / "complement_state.json", {
            "method": self.method, "archive": [{"item": e.item.to_dict(), "born": e.born}
                                                for e in self.archive.values()],
            "history": self.history, "probes": self.probes, "permanent_pairs": self.permanent_pairs})

    def run(self):
        started, status, stop = time.monotonic(), "completed", "maximum rounds reached"
        train = self.evaluator.dataset.split("train", self.search["train_limit"])
        validation = self.evaluator.dataset.split("validation", self.search["validation_limit"])
        initial, initial_cost = None, 0.0
        self.persist()
        try:
            self.best_validation = self.evaluator.evaluate(self.best, validation)
            initial, initial_cost = self.best_validation.score, self.budget.spent
            for round_id in range(int(self.search["rounds"])):
                if self.search["max_seconds"] and time.monotonic() - started >= self.search["max_seconds"]:
                    status, stop = "time_limit", "wall-clock cap reached between rounds"
                    break
                if not self._can_start_round(round_id):
                    status, stop = 'insufficient_round_reserve', 'estimated finishing reserve would be exhausted'
                    break
                self.floor(round_id, train, validation)
        except BudgetExceeded as exc:
            status, stop = "budget_exhausted", str(exc)
            self.events.emit("complement_chain_aborted", reason=stop, spent=self.budget.spent)
        finally:
            self.persist()
        unfinished = max(0.0, self.budget.spent - initial_cost -
                         sum(r["round_spent"] for r in self.history))
        return {"method": self.method, "status": status, "stop_reason": stop,
                "partial": self.best_validation is None, "test_evaluated": False,
                "initial_validation_score": initial,
                "final_validation_score": self.best_validation.score if self.best_validation else None,
                "best_build": self.best.to_dict(), "completed_rounds": len(self.history),
                "execution_changed_from_start": self.best.execution_id != self.start.execution_id,
                "accepted_rounds": sum(r["accepted"] for r in self.history),
                "initial_validation_spent": initial_cost,
                "unfinished_round_spent": unfinished,
                "unfinished_chain_gain": 0.0,
                "budget": self.budget.snapshot(), "history": self.history,
                "seconds": time.monotonic() - started,
                "train_ids": [t.id for t in train], "validation_ids": [t.id for t in validation],
                "temporary_support_boundary": "original_inputs_before_first_actor_call",
                "support_origin": "declared_bounded_input_observation_program",
                "selection_rule": {"complement": "positive_assisted_net_gain_then_specific_response",
                    "complement_total": "positive_assisted_net_gain_then_assisted_total_score",
                    "complement_random": "random_weak_consumer_with_the_same_probe_evidence",
                    "complement_direct": "random_weak_consumer_without_intervention",
                    "complement_crossover": "source_adoption_then_ordinary_reflection"}[self.method],
                "probe_count": len(self.probes), "permanent_pair_count": len(self.permanent_pairs),
                "reward_scope": "entire_paid_chain; no reward from assisted execution"}
