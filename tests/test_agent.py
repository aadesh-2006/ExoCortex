"""
Unit tests for ExoCortexAgent cognitive loop and intent planning.
"""

import unittest
from exocortex.agent import ExoCortexAgent, AgentRunResult
from exocortex.config import Settings
from exocortex.intent import IntentAnalysis, IntentCategory, IntentEngine
from exocortex.planner import AgentPlanner, StepStatus


class TestAgent(unittest.TestCase):
    def test_intent_engine(self):
        engine = IntentEngine()
        res = engine.analyze("Check the health of the system")
        self.assertEqual(res.primary_category, IntentCategory.SYSTEM_DIAGNOSTICS)

        res_hw = engine.analyze("Show hardware specs and snapdragon npu")
        self.assertEqual(res_hw.primary_category, IntentCategory.HARDWARE_INSPECTION)

    def test_planner(self):
        planner = AgentPlanner()
        intent = IntentAnalysis(
            raw_query="check health",
            primary_category=IntentCategory.SYSTEM_DIAGNOSTICS,
            confidence=0.95,
        )
        plan = planner.create_plan(intent)
        self.assertEqual(plan.total_steps, 1)
        self.assertEqual(plan.steps[0].tool_name, "health_check")

    def test_agent_run_loop(self):
        agent = ExoCortexAgent()
        result = agent.run("Show system hardware specs")
        self.assertIsInstance(result, AgentRunResult)
        self.assertTrue(result.success)
        self.assertTrue(len(result.response_text) > 0)
        self.assertTrue(result.plan.is_complete)
        self.assertEqual(len(result.executed_tools), 1)
        self.assertEqual(result.executed_tools[0]["tool"], "system_info")


if __name__ == "__main__":
    unittest.main()
