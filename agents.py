"""Agent protocol and provided Rescue Rover baselines."""

from __future__ import annotations

import random
import heapq
from typing import Protocol

from proplogic import KnowledgeBase, atom
from rover import Action, ActionKind, Percept, Position, Scanner, ScanObservation


class RoverAgent(Protocol):
    def reset(self) -> None:
        """Clear episode-specific state."""

    def act(self, percept: Percept) -> Action:
        """Choose one action from the current percept and observation history."""


class StopAgent:
    """Minimal interface example that immediately ends the mission."""

    def reset(self) -> None:
        pass

    def act(self, percept: Percept) -> Action:
        del percept
        return Action.stop()


class RandomAgent:
    """Seeded baseline that does not reason about scanner evidence."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, percept: Percept) -> Action:
        actions = self._candidate_actions(percept)
        return self._rng.choice(actions)

    def _candidate_actions(self, percept: Percept) -> list[Action]:
        actions = [Action.retrieve(), Action.stop()]
        actions.append(
            Action.disable_traction()
            if percept.traction_enabled
            else Action.enable_traction()
        )

        row = percept.position.row
        column = percept.position.column
        for candidate in (
            Position(row - 1, column),
            Position(row + 1, column),
            Position(row, column - 1),
            Position(row, column + 1),
        ):
            if 0 <= candidate.row < percept.rows and 0 <= candidate.column < percept.columns:
                actions.append(Action.move(candidate))

        for scan_row in range(percept.rows):
            for scan_column in range(percept.columns - 2):
                actions.append(
                    Action.structural_scan(
                        *(Position(scan_row, scan_column + offset) for offset in range(3))
                    )
                )
        for scan_row in range(percept.rows - 2):
            for scan_column in range(percept.columns):
                actions.append(
                    Action.structural_scan(
                        *(Position(scan_row + offset, scan_column) for offset in range(3))
                    )
                )
        for scan_row in range(percept.rows - 1):
            for scan_column in range(percept.columns - 1):
                actions.append(
                    Action.surface_scan(
                        Position(scan_row, scan_column),
                        Position(scan_row, scan_column + 1),
                        Position(scan_row + 1, scan_column),
                        Position(scan_row + 1, scan_column + 1),
                    )
                )
        return actions


class LogicAgent:
    """Required knowledge-based rover agent implemented by students."""

    def __init__(self, seed: int, use_surface_scans: bool = False) -> None:
        self.seed = seed
        self.use_surface_scans = use_surface_scans
        self.reset()

    def reset(self) -> None:
        self.knowledge = KnowledgeBase()
        self.initialized = False
        self.last_processed_action = -1
        self.proven_safe_tiles = set()

        self.pending_scans = []
        self.scan_index = 0
        self.planned_actions = []
        self.plan_index = 0
        self.mission_routes = None
        self.scanned_surface_regions = set()


    def terrain_symbol(self, terrain: str, position: Position):
        return atom(f"{terrain}_{position.row}_{position.column}")

    def initialize_knowledge(self, percept: Percept) -> None:
        for row in range(percept.rows):
            for column in range(percept.columns):
                position = Position(row, column)
                stable = self.terrain_symbol("stable", position).symbol
                slippery = self.terrain_symbol("slippery", position).symbol
                fragile = self.terrain_symbol("fragile", position).symbol

                self.knowledge.add_formula(
                    f"{stable} OR {slippery} OR {fragile}"
                )
                self.knowledge.add_formula(
                    f"NOT ({stable} AND {slippery})"
                )
                self.knowledge.add_formula(
                    f"NOT ({stable} AND {fragile})"
                )
                self.knowledge.add_formula(
                    f"NOT ({slippery} AND {fragile})"
                )

        start = self.terrain_symbol("stable", percept.position).symbol
        self.knowledge.add_formula(start)
        self.initialized = True

        self.pending_scans = self.structural_scan_actions(percept.rows, percept.columns)

    def add_scan_observation(self, observation: ScanObservation) -> None:
        terrain = (
            "fragile"
            if observation.scanner is Scanner.STRUCTURAL
            else "slippery"
        )

        symbols = [
            self.terrain_symbol(terrain, position).symbol
            for position in observation.region
        ]

        if observation.positive:
            self.knowledge.add_formula(" OR ".join(symbols))
        else:
            for symbol in symbols:
                self.knowledge.add_formula(f"NOT {symbol}")

        if not self.knowledge.is_consistent():
            raise RuntimeError(
                f"Inconsistent knowledge after scan: {observation}"
            )

    def update_knowledge(self, percept: Percept) -> None:
        if not self.initialized:
            self.initialize_knowledge(percept)

        if percept.action_count <= self.last_processed_action:
            return

        if percept.last_outcome is not None:
            observation = percept.last_outcome.scan
            if observation is not None:
                self.add_scan_observation(observation)

        self.last_processed_action = percept.action_count

    def is_non_fragile(self, position: Position) -> bool:
        fragile = self.terrain_symbol("fragile", position).symbol
        return self.knowledge.entails(f"NOT {fragile}")

    def structural_scan_actions(self, rows: int, columns: int) -> list[Action]:
        scans = []

        # Row scan, 3 tiles across.
        for row in range(rows):
            for column in range(columns - 2):
                scans.append(
                    Action.structural_scan(
                        Position(row, column),
                        Position(row, column + 1),
                        Position(row, column + 2),
                    )
                )

        # Column scan, 3 tiles up/down
        for row in range(rows - 2):
            for column in range(columns):
                scans.append(
                    Action.structural_scan(
                        Position(row, column),
                        Position(row + 1, column),
                        Position(row + 2, column),
                    )
                )

        return scans

    def surface_scan_actions(self, rows: int, columns: int) -> list[Action]:
        scans = []

        for row in range(rows - 1):
            for column in range(columns - 1):
                scans.append(
                    Action.surface_scan(
                        Position(row, column),
                        Position(row, column + 1),
                        Position(row + 1, column),
                        Position(row + 1, column + 1),
                    )
                )

        return scans

    def heuristic(self, position: Position, goal: Position) -> int:
        return (
            abs(position.row - goal.row) + abs(position.column - goal.column)
        )

    def neighbors(self, position: Position) -> list[Position]:
        return [
            Position(position.row - 1, position.column),
            Position(position.row + 1, position.column),
            Position(position.row, position.column - 1),
            Position(position.row, position.column + 1),
        ]

    def search_route(
        self,
        start: Position,
        goal: Position,
        safe_tiles: set[Position],
    ) -> list[Position] | None:
        if start not in safe_tiles or goal not in safe_tiles:
            return None

        frontier = []
        heapq.heappush(frontier, (self.heuristic(start, goal), 0, 0, start))
        best_g = {start: 0}
        came_from = {}
        tie_breaker = 1

        while frontier:
            _, _, current_g, current = heapq.heappop(frontier)

            if current_g > best_g[current]:
                continue

            if current == goal:
                route = [current]
                while current in came_from:
                    current = came_from[current]
                    route.append(current)
                route.reverse()
                return route

            for neighbor in self.neighbors(current):
                if neighbor not in safe_tiles:
                    continue

                new_g = current_g + 1

                if neighbor not in best_g or new_g < best_g[neighbor]:
                    best_g[neighbor] = new_g
                    came_from[neighbor] = current
                    priority = new_g + self.heuristic(neighbor, goal)
                    heapq.heappush(
                        frontier,
                        (priority, tie_breaker, new_g, neighbor),
                    )
                    tie_breaker += 1

        return None

    def inferred_safe_tiles(self, rows: int, columns: int) -> set[Position]:
        for row in range(rows):
            for column in range(columns):
                position = Position(row, column)
                if position in self.proven_safe_tiles:
                    continue
                if self.is_non_fragile(position):
                    self.proven_safe_tiles.add(position)

        return set(self.proven_safe_tiles)

    def routes_to_actions(
        self,
        to_module: list[Position],
        to_exit: list[Position],
        traction_enabled: bool,
    ) -> list[Action]:
        actions = []

        if not traction_enabled:
            actions.append(Action.enable_traction())

        for position in to_module[1:]:
            actions.append(Action.move(position))

        actions.append(Action.retrieve())

        for position in to_exit[1:]:
            actions.append(Action.move(position))

        actions.append(Action.stop())
        return actions

    def is_stable(self, position: Position) -> bool:
        return self.knowledge.entails(self.terrain_symbol("stable", position))

    def is_slippery(self, position: Position) -> bool:
        return self.knowledge.entails(self.terrain_symbol("slippery", position))

    def traction_plan(
        self, to_module, to_exit, stable_tiles, traction_enabled, action_limit,
    ):
        steps = (
            [Action.move(position) for position in to_module[1:]]
            + [Action.retrieve()]
            + [Action.move(position) for position in to_exit[1:]]
            + [Action.stop()]
        )
        states = {(traction_enabled, 0): (0, [])}
        for step in steps:
            next_states = {}
            for (enabled, count), (cost, actions) in states.items():
                choices = (False, True) if (
                    step.kind is ActionKind.MOVE and step.target in stable_tiles
                ) else (True,) if step.kind is ActionKind.MOVE else (enabled,)
                for target_enabled in choices:
                    addition = []
                    extra_cost = 0
                    if target_enabled != enabled:
                        addition.append(
                            Action.enable_traction() if target_enabled
                            else Action.disable_traction()
                        )
                        extra_cost += 1
                    addition.append(step)
                    if step.kind is ActionKind.MOVE:
                        extra_cost += 3 + (3 if target_enabled else 0)
                    elif step.kind is ActionKind.RETRIEVE:
                        extra_cost += 1
                    new_count = count + len(addition)
                    if new_count > action_limit:
                        continue
                    key = (target_enabled, new_count)
                    new_cost = cost + extra_cost
                    if key not in next_states or new_cost < next_states[key][0]:
                        next_states[key] = (new_cost, actions + addition)
            states = next_states
        if not states:
            return None
        cost, actions = min(states.values(), key=lambda value: (value[0], len(value[1])))
        return actions, cost

    def choose_surface_scan(self, percept, to_module, to_exit, stable_tiles):
        baseline = self.routes_to_actions(
            to_module, to_exit, percept.traction_enabled
        )
        remaining = 100 - percept.action_count
        if remaining <= len(baseline):
            return None
        current = self.traction_plan(
            to_module, to_exit, stable_tiles, percept.traction_enabled, remaining
        )
        destinations = set(to_module[1:] + to_exit[1:])
        unresolved = {
            position for position in destinations
            if position not in stable_tiles and not self.is_slippery(position)
        }
        best_scan = None
        best_saving = 0
        for scan in self.surface_scan_actions(percept.rows, percept.columns):
            if scan.region in self.scanned_surface_regions:
                continue
            candidates = unresolved.intersection(scan.region)
            if not candidates:
                continue
            optimistic = self.traction_plan(
                to_module, to_exit, stable_tiles | candidates,
                percept.traction_enabled, remaining - 1,
            )
            if optimistic is None:
                continue
            saving = current[1] - optimistic[1] - 2
            if saving > best_saving:
                best_saving = saving
                best_scan = scan
        return best_scan

    def act(self, percept: Percept) -> Action:
        self.update_knowledge(percept)

        if not self.planned_actions:
            if self.mission_routes is None:
                safe_tiles = self.inferred_safe_tiles(percept.rows, percept.columns)
                to_module = self.search_route(
                    percept.position, percept.module_position, safe_tiles
                )
                to_exit = self.search_route(
                    percept.module_position, percept.exit_position, safe_tiles
                )
                if to_module is not None and to_exit is not None:
                    self.mission_routes = (to_module, to_exit)
            else:
                to_module, to_exit = self.mission_routes

            if to_module is not None and to_exit is not None:
                if self.use_surface_scans:
                    stable_tiles = {
                        position for position in set(to_module + to_exit)
                        if self.is_stable(position)
                    }
                    scan = self.choose_surface_scan(
                        percept, to_module, to_exit, stable_tiles
                    )
                    if scan is not None:
                        self.scanned_surface_regions.add(scan.region)
                        return scan
                    plan = self.traction_plan(
                        to_module, to_exit, stable_tiles,
                        percept.traction_enabled, 100 - percept.action_count,
                    )
                    if plan is None:
                        raise RuntimeError("Mission plan exceeds remaining action budget")
                    self.planned_actions = plan[0]
                else:
                    self.planned_actions = self.routes_to_actions(
                        to_module, to_exit, percept.traction_enabled
                    )
            else:
                if self.scan_index >= len(self.pending_scans):
                    raise RuntimeError(
                        "Structural scans exhausted without both safe routes"
                    )

                action = self.pending_scans[self.scan_index]
                self.scan_index += 1
                return action

        action = self.planned_actions[self.plan_index]
        self.plan_index += 1
        return action
