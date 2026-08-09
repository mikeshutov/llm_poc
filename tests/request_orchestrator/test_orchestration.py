from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from conversation.models.conversation_models import ConversationContext
from integrations.world_time.models import WorldTime
from personalization.profile.models import UserProfile
from personalization.user_attributes.models.user_attribute_models import UserAttribute

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.agents.main_agent.agent import run_agent
from request_orchestrator.agents.main_agent.request_analysis.prompts.request_analysis_prompt import build_request_analysis_prompt
from request_orchestrator.agents.profile_management.agent import _prepare_subagent_state
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from request_orchestrator.shared.prompts.render_agent_prompt import render_agent_prompt
from test_utilities import FakeUserAttributeRepository, MockLLM, MockLLMScenario


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

    def test_profile_management_subagent_uses_profile_goal_and_raw_user_prompt(self) -> None:
        parent_state = AgentState.new(
            task='Please remember that I like pizza and eggs.',
            max_turns=10,
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            llm=MockLLM([]),
        )
        parent_state.request_analysis.goal = "Store the user's food preferences."

        subagent_state = _prepare_subagent_state(parent_state)
        prompt = build_planner_prompt(subagent_state.to_runtime_state(parent_state))
        prompt_text = render_agent_prompt(prompt)

        self.assertEqual(subagent_state.task, 'Please remember that I like pizza and eggs.')
        self.assertEqual(
            subagent_state.request_analysis.goal,
            'Review this turn for durable user attribute maintenance needs. If attribute work is needed, plan the minimal retrieval and/or update step combination required.',
        )
        self.assertIn('Latest User Prompt:', prompt_text)
        self.assertIn('Please remember that I like pizza and eggs.', prompt_text)
        self.assertEqual(
            prompt.task,
            'Review this turn for durable user attribute maintenance needs. If attribute work is needed, plan the minimal retrieval and/or update step combination required.',
        )
        self.assertEqual(prompt.latest_user_prompt, 'Please remember that I like pizza and eggs.')
        self.assertIn('attribute_type (required): Typed user-attribute key such as `food.likes`, `projects.goals`, or `technology.skills`.', prompt_text)
        self.assertNotIn('career.likes, career.dislikes', prompt_text)

    def test_request_analysis_prompt_requires_self_contained_goal_for_downstream_steps(self) -> None:
        state = AgentState.new(
            task='Can you use the jackets you suggested earlier and narrow them to waterproof options under $200?',
            max_turns=10,
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            llm=MockLLM([]),
        )

        prompt = build_request_analysis_prompt(state)
        prompt_text = render_agent_prompt(prompt)

        self.assertIn('Make the goal self-contained for downstream steps because the full conversation context will not be passed through later.', prompt_text)
        self.assertIn('Include any relevant conversation-derived constraints, continuity, entities, or references needed by downstream planning and synthesis because the full conversation context will not be passed through later.', prompt_text)

    def test_build_llm_for_stage_reuses_existing_non_chat_llm(self) -> None:
        parent_state = AgentState.new(
            task='Please remember that I like pizza.',
            max_turns=10,
            conversation_context=ConversationContext(),
            user_profile=UserProfile(),
            llm=MockLLM([]),
        )

        self.assertIs(parent_state.build_llm_for_stage(stage='planner'), parent_state.llm)

    def test_user_attribute_creation_orchestration(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goal": "Store the user's food preferences.",
          "applicable_tool_categories": ["user_attributes"],
          "requested_user_attribute_types": [],
          "requires_tools": true,
          "context_answer_confidence": 0
        }
        """

        profile_management_planner_response = """
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

        profile_management_completion_response = """
        {
          "steps": [],
          "final_answer": null,
          "needs_replan": false
        }
        """

        main_planner_response = """
        {
          "steps": [
            {
              "id": "E1",
              "plan": "Check the current stored user attributes before responding.",
              "tool": "get_user_attributes",
              "args": {
                "limit": 10,
                "is_active": true
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
          "roundtrip_summary": "Stored the user's stated food likes as a persistent user attribute and confirmed the profile state.",
          "tool_summary": {
            "used_tools": ["get_user_attributes"],
            "produced": ["current user attribute list"],
            "entities": ["pizza", "eggs"],
            "freshness": ""
          }
        }
        """

        result, llm = self._run_case(
            user_query='Please remember that I like pizza and eggs.',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[
                    profile_management_planner_response,
                    profile_management_completion_response,
                ],
                main_planner=[main_planner_response],
                synthesis=[synthesis_response],
            ),
            patchers=[
                patch(
                    'personalization.profile.service.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
                patch(
                    'request_orchestrator.shared.tool_adapter.user_attributes.create_user_attribute.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
                patch(
                    'request_orchestrator.shared.tool_adapter.user_attributes.get_user_attributes.get_user_attribute_repo',
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

        self.assertEqual(len(llm.invocations), 5)
        self.assertIn('Latest User Prompt:', llm.prompts[0] or '')
        self.assertIn('User Profile (JSON):\n\n{}', llm.prompts[0] or '')
        self.assertNotIn('Conversation Context (JSON):', llm.prompts[3] or '')
        self.assertNotIn('recent_roundtrip_tool_summaries', llm.prompts[4] or '')
        self.assertNotIn('conversation_summary', llm.prompts[4] or '')
        self.assertLessEqual((llm.prompts[4] or '').count('message_index'), 3)

    def test_requested_user_attributes_are_loaded_after_request_analysis(self) -> None:
        fake_repo = FakeUserAttributeRepository(
            created_attributes=[
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=['pizza'],
                    attribute_embedding=None,
                    attribute_type='food.likes',
                    group_key=None,
                    source='explicit',
                    is_active=True,
                    created_at='2026-08-05T00:00:00Z',
                    updated_at='2026-08-05T00:00:00Z',
                    confidence=0.9,
                    importance=0.8,
                ),
                UserAttribute(
                    id=uuid4(),
                    user_id=None,
                    value=['eggs'],
                    attribute_embedding=None,
                    attribute_type='food.likes',
                    group_key=None,
                    source='explicit',
                    is_active=True,
                    created_at='2026-08-04T00:00:00Z',
                    updated_at='2026-08-04T00:00:00Z',
                    confidence=0.85,
                    importance=0.75,
                )
            ]
        )

        request_analysis_response = """
        {
          "goal": "Use the user's food preferences to help with the request.",
          "applicable_tool_categories": ["food"],
          "requested_user_attribute_types": ["food.likes"],
          "requires_tools": true,
          "context_answer_confidence": 0
        }
        """

        profile_management_planner_response = """
        {
          "steps": [],
          "final_answer": null,
          "needs_replan": false
        }
        """

        main_planner_response = """
        {
          "steps": [
            {
              "id": "E1",
              "plan": "Look up a pizza-related meal idea.",
              "tool": "search_meals",
              "args": {
                "query": "pizza"
              }
            }
          ],
          "final_answer": null,
          "needs_replan": false
        }
        """

        synthesis_response = """
        {
          "result": ["I used your stored food preferences while looking up a meal idea."],
          "follow_up": "",
          "clarifying_question": "",
          "roundtrip_summary": "Loaded the user's stored food likes and used them while planning a meal-related response.",
          "tool_summary": {
            "used_tools": ["search_meals"],
            "produced": ["meal ideas"],
            "entities": ["pizza"],
            "freshness": ""
          }
        }
        """

        result, llm = self._run_case(
            user_query='Use what you know about my food preferences to suggest something.',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[profile_management_planner_response],
                main_planner=[main_planner_response],
                synthesis=[synthesis_response],
            ),
            patchers=[
                patch(
                    'personalization.profile.service.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
                patch(
                    'request_orchestrator.shared.tool_adapter.food.search_meals._meal_db_client.search',
                    return_value=[{'strMeal': 'Pizza Margherita'}],
                ),
            ],
        )

        self.assertEqual(result.answer, ['I used your stored food preferences while looking up a meal idea.'])
        self.assertEqual(len(llm.invocations), 4)
        self.assertIn('User Profile (JSON):\n\n{}', llm.prompts[0] or '')
        self.assertNotIn('pizza', llm.prompts[0] or '')

        main_agent_logs = result.agent_logs.get('main_agent', [])
        request_analysis_log = next(log for log in main_agent_logs if log.get('kind') == 'request_analysis')
        profile_load_log = next(log for log in main_agent_logs if log.get('kind') == 'profile_load')

        self.assertEqual(request_analysis_log.get('data', {}).get('requested_user_attribute_types'), ['food.likes'])
        self.assertEqual(profile_load_log.get('data', {}).get('loaded_attribute_count'), 1)
        self.assertEqual(profile_load_log.get('data', {}).get('loaded_attribute_types'), ['food.likes'])
        self.assertEqual(
            profile_load_log.get('data', {}).get('loaded_attributes'),
            [
                {
                    'attribute_type': 'food.likes',
                    'group_key': None,
                    'value': ['pizza', 'eggs'],
                }
            ],
        )

    def test_calculate_tool_orchestration(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goal": "Calculate the result of the math expression.",
          "applicable_tool_categories": ["math"],
          "requested_user_attribute_types": [],
          "requires_tools": true,
          "context_answer_confidence": 0
        }
        """

        profile_management_planner_response = """
        {
          "steps": [
            {
              "id": "E1",
              "plan": "Inspect the current durable user attributes before doing anything else.",
              "tool": "get_user_attributes",
              "args": {
                "limit": 10,
                "is_active": true
              }
            }
          ],
          "final_answer": null,
          "needs_replan": false
        }
        """

        profile_management_completion_response = """
        {
          "steps": [],
          "final_answer": null,
          "needs_replan": false
        }
        """

        main_planner_response = """
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
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[
                    profile_management_planner_response,
                    profile_management_completion_response,
                ],
                main_planner=[main_planner_response],
                synthesis=[synthesis_response],
            ),
            patchers=[
                patch(
                    'personalization.profile.service.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
                patch(
                    'request_orchestrator.shared.tool_adapter.user_attributes.get_user_attributes.get_user_attribute_repo',
                    return_value=fake_repo,
                )
            ],
        )

        self.assertEqual(result.answer, ['The result is 47.0.'])
        self.assertEqual(result.tool_summary.get('used_tools'), ['calculate'])
        self.assertEqual(len(llm.invocations), 5)

    def test_world_time_tool_orchestration(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goal": "Find the current time in Tokyo.",
          "applicable_tool_categories": ["calendar"],
          "requested_user_attribute_types": [],
          "requires_tools": true,
          "context_answer_confidence": 0
        }
        """

        profile_management_planner_response = """
        {
          "steps": [
            {
              "id": "E1",
              "plan": "Inspect the current durable user attributes before doing anything else.",
              "tool": "get_user_attributes",
              "args": {
                "limit": 10,
                "is_active": true
              }
            }
          ],
          "final_answer": null,
          "needs_replan": false
        }
        """

        profile_management_completion_response = """
        {
          "steps": [],
          "final_answer": null,
          "needs_replan": false
        }
        """

        main_planner_response = """
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
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[
                    profile_management_planner_response,
                    profile_management_completion_response,
                ],
                main_planner=[main_planner_response],
                synthesis=[synthesis_response],
            ),
            patchers=[
                patch(
                    'personalization.profile.service.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
                patch(
                    'request_orchestrator.shared.tool_adapter.user_attributes.get_user_attributes.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
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
        self.assertEqual(len(llm.invocations), 5)

    def test_clarifying_question_wins_over_follow_up(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goal": "Clarify an underspecified request.",
          "applicable_tool_categories": [],
          "requested_user_attribute_types": [],
          "requires_tools": false,
          "context_answer_confidence": 1
        }
        """

        profile_management_planner_response = """
        {
          "steps": [],
          "final_answer": null,
          "needs_replan": false
        }
        """

        main_planner_response = """
        {
          "steps": [],
          "final_answer": null,
          "needs_replan": false
        }
        """

        synthesis_response = """
        {
          "result": ["I can help with that."],
          "follow_up": "Do you want a short or detailed answer?",
          "clarifying_question": "Which product category do you want to focus on?",
          "roundtrip_summary": "The request was underspecified, so the response preserved only the clarifying question and dropped the follow-up variant.",
          "tool_summary": {
            "used_tools": [],
            "produced": [],
            "entities": [],
            "freshness": ""
          }
        }
        """

        result, _ = self._run_case(
            user_query='Help me pick something.',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[profile_management_planner_response],
                main_planner=[main_planner_response],
                synthesis=[synthesis_response],
            ),
            patchers=[
                patch(
                    'personalization.profile.service.get_user_attribute_repo',
                    return_value=fake_repo,
                ),
            ],
        )

        self.assertEqual(result.follow_up, '')
        self.assertEqual(result.clarifying_question, 'Which product category do you want to focus on?')
        self.assertEqual(result.next_question, 'Which product category do you want to focus on?')


if __name__ == '__main__':
    unittest.main()






