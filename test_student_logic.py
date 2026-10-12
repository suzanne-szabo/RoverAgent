

import unittest
from dataclasses import replace
from unittest.mock import patch

from agents import LogicAgent
from rover import Action, ActionOutcome, Event, Percept, Position, Scanner, ScanObservation
from rover import (
    ActionKind, MissionStatus, RescueRoverEnvironment, RoverScenario, Terrain,
)
from run_experiment import run_scenario


class ScanTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = LogicAgent(4300)
        self.region = (Position(0, 0), Position(0, 1), Position(0, 2))
        percept = Percept(
            rows=1,
            columns=4,
            position=Position(0, 3),
            module_position=Position(0, 0),
            exit_position=Position(0, 2),
            traction_enabled=False,
            carrying_module=False,
            action_count=0,
            last_outcome=None,
        )
        self.agent.initialize_knowledge(percept)

    def test_negative_structural_scan_proves_each_tile_non_fragile(self) -> None:
        observation = ScanObservation(Scanner.STRUCTURAL, self.region, False)
        self.agent.add_scan_observation(observation)

        for position in self.region:
            with self.subTest(position=position):
                self.assertTrue(self.agent.is_non_fragile(position))
                self.assertFalse(self.agent.knowledge.entails(
                    self.agent.terrain_symbol("stable", position)
                ))
                self.assertFalse(self.agent.knowledge.entails(
                    self.agent.terrain_symbol("slippery", position)
                ))
        self.assertTrue(self.agent.knowledge.is_consistent())

    def test_unobserved_tile_remains_unknown_and_excluded_from_search(self) -> None:
        position = Position(0, 0)

        for terrain in ("stable", "slippery", "fragile"):
            with self.subTest(terrain=terrain):
                symbol = self.agent.terrain_symbol(terrain, position).symbol
                self.assertFalse(self.agent.knowledge.entails(symbol))
                self.assertFalse(self.agent.knowledge.entails(f"NOT {symbol}"))

        self.assertFalse(self.agent.is_non_fragile(position))
        safe_tiles = self.agent.inferred_safe_tiles(1, 4)
        self.assertNotIn(position, safe_tiles)
        self.assertEqual(safe_tiles, {Position(0, 3)})
        self.assertTrue(self.agent.knowledge.is_consistent())

    def test_contradictory_scan_evidence_raises_error(self) -> None:
        self.agent.add_scan_observation(
            ScanObservation(Scanner.STRUCTURAL, self.region, False)
        )
        self.assertTrue(self.agent.knowledge.is_consistent())

        # The same fixed region cannot be both entirely non-fragile and
        # contain at least one fragile tile.
        with self.assertRaisesRegex(RuntimeError, "Inconsistent knowledge after scan"):
            self.agent.add_scan_observation(
                ScanObservation(Scanner.STRUCTURAL, self.region, True)
            )

    def test_positive_structural_scan_proves_disjunction_only(self) -> None:
        observation = ScanObservation(Scanner.STRUCTURAL, self.region, True)
        self.agent.add_scan_observation(observation)

        self.assertTrue(self.agent.knowledge.entails(
            "fragile_0_0 OR fragile_0_1 OR fragile_0_2"
        ))
        for position in self.region:
            with self.subTest(position=position):
                self.assertFalse(self.agent.knowledge.entails(
                    self.agent.terrain_symbol("fragile", position)
                ))
                self.assertFalse(self.agent.is_non_fragile(position))
        self.assertTrue(self.agent.knowledge.is_consistent())


class EpisodeStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = LogicAgent(4300, use_surface_scans=True)
        self.initial_percept = Percept(
            rows=1,
            columns=4,
            position=Position(0, 3),
            module_position=Position(0, 0),
            exit_position=Position(0, 2),
            traction_enabled=False,
            carrying_module=False,
            action_count=0,
            last_outcome=None,
        )

    def test_repeated_percept_processes_scan_only_once(self) -> None:
        self.agent.update_knowledge(self.initial_percept)
        observation = ScanObservation(
            Scanner.STRUCTURAL,
            (Position(0, 0), Position(0, 1), Position(0, 2)),
            False,
        )
        percept = replace(
            self.initial_percept,
            action_count=1,
            last_outcome=ActionOutcome(True, Event.SCANNED, -2, observation),
        )

        with patch.object(
            self.agent, "add_scan_observation",
            wraps=self.agent.add_scan_observation,
        ) as translate:
            self.agent.update_knowledge(percept)
            clauses = self.agent.knowledge.clauses
            self.agent.update_knowledge(percept)
            translate.assert_called_once_with(observation)
            self.assertEqual(self.agent.knowledge.clauses, clauses)

        self.assertEqual(self.agent.last_processed_action, 1)
        self.assertTrue(self.agent.is_non_fragile(Position(0, 0)))

    def test_reset_clears_episode_state_and_allows_fresh_initialization(self) -> None:
        self.agent.update_knowledge(self.initial_percept)
        old_knowledge = self.agent.knowledge
        self.agent.knowledge.add_formula("NOT fragile_0_0")
        self.agent.inferred_safe_tiles(1, 4)
        self.agent.last_processed_action = 7
        self.agent.scan_index = 2
        self.agent.planned_actions = [Action.enable_traction(), Action.stop()]
        self.agent.plan_index = 1

        self.agent.reset()

        self.assertIsNot(self.agent.knowledge, old_knowledge)
        self.assertEqual(self.agent.knowledge.clauses, ())
        self.assertFalse(self.agent.initialized)
        self.assertEqual(self.agent.last_processed_action, -1)
        self.assertEqual(self.agent.proven_safe_tiles, set())
        self.assertEqual(self.agent.pending_scans, [])
        self.assertEqual(self.agent.scan_index, 0)
        self.assertEqual(self.agent.planned_actions, [])
        self.assertEqual(self.agent.plan_index, 0)
        self.assertEqual(self.agent.seed, 4300)
        self.assertTrue(self.agent.use_surface_scans)

        self.agent.update_knowledge(self.initial_percept)
        self.assertTrue(self.agent.initialized)
        self.assertFalse(self.agent.is_non_fragile(Position(0, 0)))
        self.assertEqual(
            self.agent.inferred_safe_tiles(1, 4), {Position(0, 3)}
        )


class GraphSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = LogicAgent(4300)
        self.start = Position(0, 0)
        self.goal = Position(0, 2)
        self.agent.initialize_knowledge(Percept(
            rows=3,
            columns=3,
            position=self.start,
            module_position=self.goal,
            exit_position=Position(1, 2),
            traction_enabled=False,
            carrying_module=False,
            action_count=0,
            last_outcome=None,
        ))

    def test_astar_follows_shortest_route_through_inferred_safe_tiles(self) -> None:
        # At least one top-row tile is fragile.
        self.agent.add_scan_observation(ScanObservation(
            Scanner.STRUCTURAL,
            (Position(0, 0), Position(0, 1), Position(0, 2)),
            True,
        ))
        self.agent.add_scan_observation(ScanObservation(
            Scanner.STRUCTURAL,
            (Position(1, 0), Position(1, 1), Position(1, 2)),
            False,
        ))
        self.agent.add_scan_observation(ScanObservation(
            Scanner.STRUCTURAL,
            (Position(0, 2), Position(1, 2), Position(2, 2)),
            False,
        ))

        safe_tiles = self.agent.inferred_safe_tiles(3, 3)
        self.assertTrue(self.agent.knowledge.entails("fragile_0_1"))
        self.assertNotIn(Position(0, 1), safe_tiles)
        route = self.agent.search_route(self.start, self.goal, safe_tiles)
        self.assertEqual(route, [
            Position(0, 0), Position(1, 0), Position(1, 1),
            Position(1, 2), Position(0, 2),
        ])
        self.assertTrue(all(position in safe_tiles for position in route))
        self.assertTrue(self.agent.knowledge.is_consistent())

    def test_astar_returns_none_when_goal_is_unknown(self) -> None:
        safe_tiles = self.agent.inferred_safe_tiles(3, 3)
        self.assertEqual(safe_tiles, {self.start})
        self.assertIsNone(self.agent.search_route(self.start, self.goal, safe_tiles))

    def test_astar_returns_none_for_disconnected_proven_safe_goal(self) -> None:
        self.agent.add_scan_observation(ScanObservation(
            Scanner.STRUCTURAL,
            (Position(0, 2), Position(1, 2), Position(2, 2)),
            False,
        ))
        safe_tiles = self.agent.inferred_safe_tiles(3, 3)
        self.assertIn(self.goal, safe_tiles)
        self.assertNotIn(Position(0, 1), safe_tiles)
        self.assertNotIn(Position(1, 0), safe_tiles)
        self.assertIsNone(self.agent.search_route(self.start, self.goal, safe_tiles))


class RouteActionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = LogicAgent(4300)
        self.to_module = [Position(0, 0), Position(0, 1), Position(1, 1)]
        self.to_exit = [Position(1, 1), Position(1, 2), Position(0, 2)]

    def test_routes_become_ordered_mission_actions(self) -> None:
        actions = self.agent.routes_to_actions(
            self.to_module, self.to_exit, traction_enabled=False
        )
        self.assertEqual(actions, [
            Action.enable_traction(),
            Action.move(Position(0, 1)),
            Action.move(Position(1, 1)),
            Action.retrieve(),
            Action.move(Position(1, 2)),
            Action.move(Position(0, 2)),
            Action.stop(),
        ])
        # Conversion must not modify the routes returned by search.
        self.assertEqual(self.to_module, [
            Position(0, 0), Position(0, 1), Position(1, 1),
        ])
        self.assertEqual(self.to_exit, [
            Position(1, 1), Position(1, 2), Position(0, 2),
        ])

    def test_enabled_traction_does_not_produce_redundant_toggle(self) -> None:
        actions = self.agent.routes_to_actions(
            self.to_module, self.to_exit, traction_enabled=True
        )
        self.assertEqual(actions, [
            Action.move(Position(0, 1)),
            Action.move(Position(1, 1)),
            Action.retrieve(),
            Action.move(Position(1, 2)),
            Action.move(Position(0, 2)),
            Action.stop(),
        ])


class MissionPreconditionTests(unittest.TestCase):
    def test_agent_retrieves_at_module_and_stops_carrying_at_exit(self) -> None:
        scenario = RoverScenario(
            identifier="student-preconditions",
            terrain=((
                Terrain.STABLE, Terrain.SLIPPERY,
                Terrain.STABLE, Terrain.SLIPPERY,
            ),),
            start=Position(0, 0),
            module=Position(0, 2),
            exit=Position(0, 3),
        )
        environment = RescueRoverEnvironment(scenario)
        agent = LogicAgent(4300)
        retrieval_count = 0
        stop_count = 0

        for _ in range(100):
            if environment.status is not MissionStatus.RUNNING:
                break

            percept = environment.percept()
            action = agent.act(percept)
            if action.kind is ActionKind.RETRIEVE:
                retrieval_count += 1
                self.assertEqual(percept.position, percept.module_position)
                self.assertFalse(percept.carrying_module)
            elif action.kind is ActionKind.STOP:
                stop_count += 1
                self.assertEqual(percept.position, percept.exit_position)
                self.assertTrue(percept.carrying_module)

            transition = environment.step(action)
            self.assertTrue(transition.outcome.succeeded)

        self.assertEqual(retrieval_count, 1)
        self.assertEqual(stop_count, 1)
        self.assertIs(environment.status, MissionStatus.SUCCESS)
        self.assertEqual(environment.invalid_actions, 0)
        self.assertEqual(environment.unprotected_slips, 0)


class ReproducibilityTests(unittest.TestCase):
    def test_same_scenario_and_seed_reproduce_complete_result(self) -> None:
        scenario = RoverScenario(
            identifier="student-reproducibility",
            terrain=((
                Terrain.STABLE, Terrain.SLIPPERY,
                Terrain.STABLE, Terrain.SLIPPERY,
            ),),
            start=Position(0, 0),
            module=Position(0, 2),
            exit=Position(0, 3),
        )

        first = run_scenario(LogicAgent, scenario, seed=4300)
        second = run_scenario(LogicAgent, scenario, seed=4300)

        self.assertIs(first.status, MissionStatus.SUCCESS)
        self.assertTrue(first.trace)
        self.assertEqual(first, second)


class OverlappingScanTests(unittest.TestCase):
    def test_overlapping_structural_scans_identify_fragile_tile(self) -> None:
        agent = LogicAgent(4300)
        percept = Percept(
            rows=1,
            columns=5,
            position=Position(0, 4),
            module_position=Position(0, 1),
            exit_position=Position(0, 3),
            traction_enabled=False,
            carrying_module=False,
            action_count=0,
            last_outcome=None,
        )
        agent.initialize_knowledge(percept)

        
        agent.add_scan_observation(ScanObservation(
            Scanner.STRUCTURAL,
            (Position(0, 0), Position(0, 1), Position(0, 2)),
            True,
        ))
        self.assertFalse(agent.knowledge.entails("fragile_0_0"))

        agent.add_scan_observation(ScanObservation(
            Scanner.STRUCTURAL,
            (Position(0, 1), Position(0, 2), Position(0, 3)),
            False,
        ))

        self.assertTrue(agent.knowledge.entails("fragile_0_0"))
        self.assertFalse(agent.is_non_fragile(Position(0, 0)))
        for column in (1, 2, 3):
            with self.subTest(column=column):
                self.assertTrue(agent.is_non_fragile(Position(0, column)))
        self.assertTrue(agent.knowledge.is_consistent())


class SurfaceScanTranslationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = LogicAgent(4300)
        self.region = (
            Position(0, 0), Position(0, 1),
            Position(1, 0), Position(1, 1),
        )
        
        percept = Percept(
            rows=2,
            columns=3,
            position=Position(0, 2),
            module_position=Position(0, 0),
            exit_position=Position(1, 1),
            traction_enabled=False,
            carrying_module=False,
            action_count=0,
            last_outcome=None,
        )
        self.agent.initialize_knowledge(percept)

    def test_negative_surface_scan_proves_each_tile_not_slippery(self) -> None:
        observation = ScanObservation(Scanner.SURFACE, self.region, False)
        self.agent.add_scan_observation(observation)

        for position in self.region:
            with self.subTest(position=position):
                slippery = self.agent.terrain_symbol("slippery", position).symbol
                self.assertTrue(self.agent.knowledge.entails(f"NOT {slippery}"))
                # Not slippery still allows stable or fragile terrain.
                self.assertFalse(self.agent.is_non_fragile(position))
                self.assertFalse(self.agent.knowledge.entails(
                    self.agent.terrain_symbol("stable", position)
                ))
        self.assertTrue(self.agent.knowledge.is_consistent())

    def test_positive_surface_scan_proves_disjunction_only(self) -> None:
        observation = ScanObservation(Scanner.SURFACE, self.region, True)
        self.agent.add_scan_observation(observation)

        self.assertTrue(self.agent.knowledge.entails(
            "slippery_0_0 OR slippery_0_1 OR slippery_1_0 OR slippery_1_1"
        ))
        for position in self.region:
            with self.subTest(position=position):
                slippery = self.agent.terrain_symbol("slippery", position).symbol
                self.assertFalse(self.agent.knowledge.entails(slippery))
                self.assertFalse(self.agent.knowledge.entails(f"NOT {slippery}"))
        self.assertTrue(self.agent.knowledge.is_consistent())


class SurfacePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = LogicAgent(4300, use_surface_scans=True)
        self.percept = Percept(
            2, 3, Position(0, 0), Position(0, 2), Position(1, 2),
            False, False, 0, None,
        )
        self.agent.initialize_knowledge(self.percept)
        self.to_module = [Position(0, 0), Position(0, 1), Position(0, 2)]
        self.to_exit = [Position(0, 2), Position(1, 2)]

    def test_terrain_queries_distinguish_stable_slippery_and_unknown(self) -> None:
        self.agent.knowledge.add_formula("slippery_0_1")
        self.assertTrue(self.agent.is_stable(Position(0, 0)))
        self.assertFalse(self.agent.is_slippery(Position(0, 0)))
        self.assertTrue(self.agent.is_slippery(Position(0, 1)))
        self.assertFalse(self.agent.is_stable(Position(0, 1)))
        self.assertFalse(self.agent.is_stable(Position(0, 2)))
        self.assertFalse(self.agent.is_slippery(Position(0, 2)))

    def test_stable_route_avoids_traction(self) -> None:
        actions, _ = self.agent.traction_plan(
            self.to_module, self.to_exit,
            set(self.to_module + self.to_exit), False, 100,
        )
        self.assertFalse(any(action.kind in (
            ActionKind.ENABLE_TRACTION, ActionKind.DISABLE_TRACTION,
        ) for action in actions))
        self.assertEqual(len(actions), 5)

    def test_tight_budget_keeps_traction_on_instead_of_extra_toggles(self) -> None:
        actions, _ = self.agent.traction_plan(
            self.to_module, self.to_exit, {Position(0, 2)}, False, 6,
        )
        self.assertEqual(actions, self.agent.routes_to_actions(
            self.to_module, self.to_exit, False,
        ))

    def test_scan_requires_potential_savings_and_spare_action(self) -> None:
        stable = {Position(0, 0)}
        scan = self.agent.choose_surface_scan(
            self.percept, self.to_module, self.to_exit, stable,
        )
        self.assertIsNotNone(scan)
        self.assertIs(scan.kind, ActionKind.SURFACE_SCAN)
        self.assertIsNone(self.agent.choose_surface_scan(
            replace(self.percept, action_count=94),
            self.to_module, self.to_exit, stable,
        ))
        self.assertIsNone(self.agent.choose_surface_scan(
            self.percept, self.to_module, self.to_exit,
            set(self.to_module + self.to_exit),
        ))

    def test_optional_policy_completes_before_limit_with_all_scans_before_moves(self) -> None:
        for slippery in (False, True):
            with self.subTest(slippery=slippery):
                terrain = Terrain.SLIPPERY if slippery else Terrain.STABLE
                scenario = RoverScenario(
                    "student-surface-policy",
                    ((Terrain.STABLE, terrain, Terrain.STABLE),
                     (Terrain.STABLE, Terrain.STABLE, Terrain.STABLE)),
                    Position(0, 0), Position(0, 2), Position(1, 2),
                )
                basic = run_scenario(LogicAgent, scenario, 4300)
                result = run_scenario(
                    lambda seed: LogicAgent(seed, use_surface_scans=True),
                    scenario, 4300,
                )
                self.assertIs(result.status, MissionStatus.SUCCESS)
                self.assertEqual(result.unprotected_slips, 0)
                self.assertEqual(result.invalid_actions, 0)
                self.assertLessEqual(result.actions, 100)
                kinds = [row.action.kind for row in result.trace]
                self.assertIn(ActionKind.SURFACE_SCAN, kinds)
                first_move = kinds.index(ActionKind.MOVE)
                self.assertFalse(any(kind in (
                    ActionKind.STRUCTURAL_SCAN, ActionKind.SURFACE_SCAN,
                ) for kind in kinds[first_move:]))
                if not slippery:
                    self.assertGreater(result.score, basic.score)
        self.agent.mission_routes = (self.to_module, self.to_exit)
        self.agent.scanned_surface_regions.add(tuple(self.to_module))
        self.agent.reset()
        self.assertIsNone(self.agent.mission_routes)
        self.assertEqual(self.agent.scanned_surface_regions, set())


if __name__ == "__main__":
    unittest.main()
