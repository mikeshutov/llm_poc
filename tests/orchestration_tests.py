from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from conversation.models.conversation_models import ConversationContext
from integrations.world_time.models import WorldTime
from personalization.profile.models import UserProfile
from request_orchestrator.agents.main_agent.agent import run_agent
from test_utilities import FakeUserAttributeRepository, MockLLM

# Simple orchestration tests to test that when the LLM responds correctly the right tools get called
# Really just making sure that things are hooked up correctly.
class MainAgentOrchestrationTest(unittest.TestCase):
    def _run_case(
        self,
        *,
        user_query: str,
        llm_responses: list[str],
        patchers: list,
        user_profile: UserProfile | None = None,
    ):
        llm = MockLLM(llm_responses)

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            result = run_agent(
                conversation_context=ConversationContext(),
                user_query=user_query,
                conversation_id='test-thread',
                user_profile=UserProfile() if user_profile is None else user_profile,
                llm=llm,
            )

        return result, llm

    def test_user_attribute_creation_orchestration(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goal": "Store the user's food preferences.",
          "applicable_tool_categories": ["user_attributes"],
          "requires_tools": true,
          "context_answer_confidence": 0
        }
        """

        planner_response = """
        {
          "steps": [
            {
              "id": "E1",
              "plan": "Store the user's food likes as a durable user attribute.",
              "tool": "create_user_attribute",
              "args": {
                "value": ["pizza", "eggs"],
                "attribute_type": "food.likes",
                "source": "explicit"
              }
            }
          ],
          "final_answer": null,
          "needs_replan": false
        }
        """

        synthesis_response = """
        {
          "result": ["Stored your food likes as a user attribute."],
          "follow_up": "",
          "clarifying_question": "",
          "roundtrip_summary": "Stored the user's stated food likes as a persistent user attribute using the user attributes tool.",
          "tool_summary": {
            "used_tools": ["create_user_attribute"],
            "produced": ["user attribute record"],
            "entities": ["pizza", "eggs"],
            "freshness": ""
          }
        }
        """

        result, llm = self._run_case(
            user_query='Please remember that I like pizza and eggs.',
            llm_responses=[
                request_analysis_response,
                planner_response,
                synthesis_response,
            ],
            patchers=[
                patch(
                    'request_orchestrator.shared.tool_adapter.user_attributes.create_user_attribute.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
                patch(
                    'request_orchestrator.shared.tool_adapter.user_attributes.create_user_attribute.embed_text',
                    return_value=[0.1, 0.2, 0.3],
                ),
            ],
        )

        self.assertEqual(result.answer, ['Stored your food likes as a user attribute.'])
        self.assertEqual(len(fake_repo.created_attributes), 1)

        created_attribute = fake_repo.created_attributes[0]
        self.assertEqual(created_attribute.value, ['pizza', 'eggs'])
        self.assertEqual(created_attribute.attribute_type, 'food.likes')
        self.assertEqual(created_attribute.source, 'explicit')

        self.assertEqual(len(llm.invocations), 3)
        self.assertIn('Latest User Prompt:', llm.prompts[0] or '')

    def test_calculate_tool_orchestration(self) -> None:
        request_analysis_response = """
        {
          "goal": "Calculate the result of the math expression.",
          "applicable_tool_categories": ["math"],
          "requires_tools": true,
          "context_answer_confidence": 0
        }
        """

        planner_response = """
        {
          "steps": [
            {
              "id": "E1",
              "plan": "Evaluate the requested expression.",
              "tool": "calculate",
              "args": {
                "expression": "(15 * 8) / 3 + 7"
              }
            }
          ],
          "final_answer": null,
          "needs_replan": false
        }
        """

        synthesis_response = """
        {
          "result": ["The result is 47.0."],
          "follow_up": "",
          "clarifying_question": "",
          "roundtrip_summary": "Calculated the requested expression using the math tool and returned the numeric result.",
          "tool_summary": {
            "used_tools": ["calculate"],
            "produced": ["numeric result"],
            "entities": ["(15 * 8) / 3 + 7"],
            "freshness": ""
          }
        }
        """

        result, llm = self._run_case(
            user_query='What is (15 * 8) / 3 + 7?',
            llm_responses=[
                request_analysis_response,
                planner_response,
                synthesis_response,
            ],
            patchers=[],
        )

        self.assertEqual(result.answer, ['The result is 47.0.'])
        self.assertEqual(result.tool_summary.get('used_tools'), ['calculate'])
        self.assertEqual(len(llm.invocations), 3)

    def test_world_time_tool_orchestration(self) -> None:
        request_analysis_response = """
        {
          "goal": "Find the current time in Tokyo.",
          "applicable_tool_categories": ["calendar"],
          "requires_tools": true,
          "context_answer_confidence": 0
        }
        """

        planner_response = """
        {
          "steps": [
            {
              "id": "E1",
              "plan": "Look up the current time in the requested timezone.",
              "tool": "get_world_time",
              "args": {
                "timezone": "Asia/Tokyo"
              }
            }
          ],
          "final_answer": null,
          "needs_replan": false
        }
        """

        synthesis_response = """
        {
          "result": ["The current time in Tokyo is 2026-08-04T21:30:00+09:00."],
          "follow_up": "",
          "clarifying_question": "",
          "roundtrip_summary": "Looked up the current time in Tokyo using the world time tool and returned the reported local datetime.",
          "tool_summary": {
            "used_tools": ["get_world_time"],
            "produced": ["local datetime", "UTC offset"],
            "entities": ["Asia/Tokyo"],
            "freshness": "current as of 2026-08-04T21:30:00+09:00"
          }
        }
        """

        result, llm = self._run_case(
            user_query='What time is it in Tokyo right now?',
            llm_responses=[
                request_analysis_response,
                planner_response,
                synthesis_response,
            ],
            patchers=[
                patch(
                    'request_orchestrator.shared.tool_adapter.calendar.world_time._client.get_time',
                    return_value=WorldTime(
                        timezone='Asia/Tokyo',
                        datetime='2026-08-04T21:30:00+09:00',
                        utc_offset='+09:00',
                        day_of_week=2,
                        abbreviation='JST',
                    ),
                )
            ],
        )

        self.assertEqual(result.answer, ['The current time in Tokyo is 2026-08-04T21:30:00+09:00.'])
        self.assertEqual(result.tool_summary.get('used_tools'), ['get_world_time'])
        self.assertEqual(len(llm.invocations), 3)


if __name__ == '__main__':
    unittest.main()
