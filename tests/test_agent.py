"""
Unit tests for ExoCortexAgent cognitive loop and structured tool-calling.
"""

import unittest
from exocortex.agent import ExoCortexAgent, AgentRunResult
from exocortex.config import Settings
from exocortex.schema import AgentDecision
from exocortex.slm.mock_provider import MockSLMProvider


class TestAgent(unittest.TestCase):
    def test_agent_run_hardware_query(self):
        agent = ExoCortexAgent()
        result = agent.run("Show system hardware specs and CPU")
        self.assertIsInstance(result, AgentRunResult)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.decision)
        self.assertIn("system_info", [s.tool for s in result.decision.steps])
        self.assertEqual(len(result.executed_tools), 1)
        self.assertEqual(result.executed_tools[0]["tool"], "system_info")
        self.assertTrue(result.executed_tools[0]["success"])

    def test_agent_run_health_query(self):
        agent = ExoCortexAgent()
        result = agent.run("Check system health status")
        self.assertTrue(result.success)
        self.assertEqual(result.decision.intent, "system_health")
        self.assertIn("health_check", [s.tool for s in result.decision.steps])

    def test_agent_run_conversational_no_tool(self):
        agent = ExoCortexAgent()
        result = agent.run("Hello! Who are you?")
        self.assertTrue(result.success)
        self.assertEqual(len(result.decision.steps), 0)
        self.assertIsNotNone(result.response_text)
        self.assertIn("ExoCortex", result.response_text)


if __name__ == "__main__":
    unittest.main()
