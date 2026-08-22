from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from common.logging import fetch_agent_logs_for_roundtrip
from conversation.models.conversation_models import ConversationContext
from conversation.models.conversation_models import ConversationEvent
from conversation.models.conversation_models import RecentRoundtrip
from integrations.world_time.models import WorldTime
from personalization.profile.models import UserProfile
from personalization.tone.models import TonePreferences
from personalization.user_attributes.models.user_attribute_models import UserAttribute

if 'yfinance' not in sys.modules:
    sys.modules['yfinance'] = ModuleType('yfinance')

if 'pycountry' not in sys.modules:
    pycountry_module = ModuleType('pycountry')
    pycountry_module.countries = SimpleNamespace(lookup=lambda value: SimpleNamespace(alpha_2=str(value).upper()))
    sys.modules['pycountry'] = pycountry_module

from request_orchestrator.agents.main_agent.profile import MAIN_AGENT_PROFILE
from request_orchestrator.agents.profile_management.profile import build_profile_management_profile
from request_orchestrator.agents.profile_management.profile import PROFILE_MANAGEMENT_PROFILE
from request_orchestrator.agents.models.user_agent import UserAgent
from request_orchestrator.orchestrator import run_agent
from request_orchestrator.models.agent_execution_context import AgentExecutionContext
from request_orchestrator.models.agent_prompt import AgentPrompt, EvidenceStep, PromptSectionKeys
from request_orchestrator.models.agent_result import AgentResult
from request_orchestrator.models.agent_state import AgentState
from request_orchestrator.models.request_analysis import RequestAnalysis, RequestAnalysisGoal
from request_orchestrator.models.evidence import EvidenceView, HydratedEvidence, ToolResult
from request_orchestrator.models.main_state import MainState
from request_orchestrator.models.plan import Plan
from request_orchestrator.models.plan_step_ids import namespace_step_id
from request_orchestrator.shared.planner.prompts.planner_prompt import build_planner_prompt
from request_orchestrator.shared.runtime_context import bind_runtime_context
from request_orchestrator.shared.request_analysis.prompts.request_analysis_prompt import build_request_analysis_prompt
from request_orchestrator.shared.evidence import (
    build_evidence_bundle_from_tool_results,
    build_evidence_steps_from_tool_results,
)
from request_orchestrator.shared.evaluator.prompts.evaluator_prompt import build_evaluator_prompt
from llm.chat_models import build_llm_for_stage
from request_orchestrator.shared.synthesis.prompts.synthesis_prompt import build_synthesis_prompt
from test_utilities import FakeUserAttributeRepository, MockLLM, MockLLMScenario


class InMemoryConversationEventRepo:
    def __init__(self) -> None:
        self.events: list[ConversationEvent] = []
        self._next_id = 1

    def create_conversation_event(self, **kwargs):
        event = ConversationEvent(
            id=self._next_id,
            conversation_id=kwargs["conversation_id"],
            roundtrip_id=kwargs.get("roundtrip_id"),
            event_type=kwargs["event_type"],
            source=kwargs["source"],
            agent_name=kwargs.get("agent_name", ""),
            node_name=kwargs.get("node_name", ""),
            step_id=kwargs.get("step_id", ""),
            iteration=kwargs.get("iteration"),
            payload=kwargs.get("payload", {}),
            created_at="2026-08-14T12:00:00Z",
        )
        self._next_id += 1
        self.events.append(event)
        return event

    def list_conversation_events_for_roundtrip(self, roundtrip_id):
        return [
            event
            for event in self.events
            if event.roundtrip_id == roundtrip_id
        ]

    def create_roundtrip_prompt(self, *args, **kwargs):
        return None


class FakeUserAgentRepository:
    def __init__(self, agents: list[UserAgent] | None = None) -> None:
        self.agents = list(agents or [])

    def list_for_user(self, user_id: str, *, is_active: bool | None = True) -> list[UserAgent]:
        if is_active is None:
            return list(self.agents)
        return [agent for agent in self.agents if agent.is_active == is_active]


def _agent_profiles_for(user_profile: UserProfile) -> list:
    return [
        build_profile_management_profile(user_profile),
        MAIN_AGENT_PROFILE,
    ]


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
        repo = InMemoryConversationEventRepo()
        conversation_id = str(uuid4())
        roundtrip_id = str(uuid4())
        resolved_user_profile = UserProfile() if user_profile is None else user_profile
        if not resolved_user_profile.user_id:
            resolved_user_profile.user_id = "test-user"

        with ExitStack() as stack:
            stack.enter_context(patch('common.logging.conversation_event_logger.get_conversation_repo', return_value=repo))
            stack.enter_context(patch('common.logging.conversation_event_view.get_conversation_repo', return_value=repo))
            stack.enter_context(
                patch(
                    'request_orchestrator.shared.agents.load_user_agents.get_user_agent_repo',
                    return_value=FakeUserAgentRepository(),
                )
            )
            for patcher in patchers:
                stack.enter_context(patcher)
            main_state = MainState.new(
                task=user_query,
                execution_context=AgentExecutionContext.new(
                    conversation_context=ConversationContext(),
                    conversation_id=conversation_id,
                    user_profile=resolved_user_profile,
                ),
                llm=llm,
                agent_profiles=_agent_profiles_for(resolved_user_profile),
            )
            with bind_runtime_context(
                conversation_id=conversation_id,
                conversation_model_config=main_state.execution_context.model_config,
                roundtrip_id=roundtrip_id,
                user_id=resolved_user_profile.user_id,
            ):
                result = run_agent(main_state)

        return result, llm, repo, roundtrip_id

    def test_profile_management_subagent_uses_profile_goal_and_raw_user_prompt(self) -> None:
        parent_state = AgentState.new(
            task='Please remember that I like pizza and eggs.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profile=MAIN_AGENT_PROFILE,
        )
        profile_state = AgentState.new(
            task=parent_state.inputs.task,
            execution_context=parent_state.execution_context,
            llm=parent_state.llm,
            agent_profile=build_profile_management_profile(parent_state.execution_context.user_profile),
        )
        profile_state.inputs.tool_category_names = sorted(profile_state.agent_profile.allowed_categories)
        prompt = build_planner_prompt(profile_state)
        prompt_text = prompt.build()

        self.assertEqual(profile_state.inputs.task, 'Please remember that I like pizza and eggs.')
        self.assertEqual(profile_state.agent_profile.max_turns, 5)
        self.assertIn('ROLE / RULES', prompt_text)
        self.assertIn('INPUT', prompt_text)
        self.assertIn('"latest_user_prompt": "Please remember that I like pizza and eggs."', prompt_text)
        self.assertIn('Please remember that I like pizza and eggs.', prompt_text)
        self.assertNotIn('Use recent_roundtrip_tool_summaries', prompt_text)
        self.assertNotIn('Use recent_roundtrips when the user refers', prompt_text)
        self.assertNotIn('Use the older string tool_summary only as fallback context', prompt_text)
        self.assertNotIn('Utilize multiple tools when it is appropriate to get full context.', prompt_text)
        self.assertNotIn('Evidence references must be defined before use.', prompt_text)
        self.assertIn("Do not make one planned tool step depend on another step's output.", prompt_text)
        self.assertIn('You may use already-available tool results from previous work', prompt_text)
        self.assertEqual(prompt.latest_user_prompt, 'Please remember that I like pizza and eggs.')
        self.assertEqual(prompt.task, 'Please remember that I like pizza and eggs.')
        self.assertIn('attribute_type (required): Typed user-attribute key such as `food.likes`, `projects.goals`, or `technology.skills`.', prompt_text)
        self.assertIn('Available attribute prefixes:', prompt_text)
        self.assertIn('Available attribute suffixes:', prompt_text)
        self.assertIn('Requested or updated attribute types must use the format prefix.suffix such as food.likes or projects.goals.', prompt_text)
        self.assertNotIn('career.likes, career.dislikes', prompt_text)


    def test_main_agent_planner_prompt_includes_request_analysis_goal(self) -> None:
        goal = 'Search the web for frozen or dry okonomiyaki kits for sale, since the user clarified they want okonomiyaki and wants a broader web check.'
        state = AgentState.new(
            task=goal,
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        prompt = build_planner_prompt(state)
        prompt_text = prompt.build()

        self.assertIn(f'"task": "{goal}"', prompt_text)
        self.assertIn(goal, prompt_text)
        self.assertNotIn('"latest_user_prompt":', prompt_text)

    def test_tone_is_included_only_for_planner_and_synthesis_prompts(self) -> None:
        profile = UserProfile(
            user_id='user-123',
            tone=TonePreferences(
                verbosity='concise',
                directness='high',
            ),
        )
        state = AgentState.new(
            task='Help me with this request.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=profile,
            ),
            llm=MockLLM([]),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        planner_prompt = build_planner_prompt(state).build()
        synthesis_prompt = build_synthesis_prompt(
            evidence=[
                EvidenceStep(
                    type='web_search_results',
                    evidence=[
                        EvidenceView(
                            evidence_id="P1E1R1",
                            item_id="known-result",
                            title='Known Result',
                            summary='Known evidence result.',
                        )
                    ],
                )
            ],
            state=state,
        ).build()
        request_analysis_prompt = build_request_analysis_prompt(
            MainState.new(
                task=state.inputs.task,
                execution_context=state.execution_context,
                llm=state.llm,
                agent_profiles=_agent_profiles_for(state.execution_context.user_profile),
            )
        ).build()
        evaluator_prompt = build_evaluator_prompt(
            state=state,
            evidence=[
                EvidenceStep(
                    type='web_search_results',
                    evidence=[
                        EvidenceView(
                            evidence_id="P1E1R1",
                            item_id="known-result",
                            title='Known Result',
                            summary='Known evidence result.',
                        )
                    ],
                )
            ],
        ).build()

        self.assertIn('"tone"', planner_prompt)
        self.assertIn('"verbosity": "concise"', planner_prompt)
        self.assertIn('"directness": "high"', planner_prompt)

        self.assertIn('"tone"', synthesis_prompt)
        self.assertIn('"verbosity": "concise"', synthesis_prompt)
        self.assertIn('"directness": "high"', synthesis_prompt)

        self.assertNotIn('"tone"', request_analysis_prompt)
        self.assertNotIn('"verbosity": "concise"', request_analysis_prompt)
        self.assertNotIn('"directness": "high"', request_analysis_prompt)

        self.assertNotIn('"tone"', evaluator_prompt)
        self.assertNotIn('"verbosity": "concise"', evaluator_prompt)
        self.assertNotIn('"directness": "high"', evaluator_prompt)

    def test_request_analysis_prompt_requires_self_contained_goal_for_downstream_steps(self) -> None:
        state = MainState.new(
            task='Can you use the jackets you suggested earlier and narrow them to waterproof options under $200?',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profiles=_agent_profiles_for(UserProfile()),
        )

        prompt = build_request_analysis_prompt(state)
        prompt_text = prompt.build()

        self.assertIn('Each goal should capture the actual objective plus any relevant conversation-derived constraints, references, or continuity needed for planning and synthesis.', prompt_text)
        self.assertIn('Name the concrete topic, subject, entity, or item in each goal instead of using vague placeholders like topic, subject, it, them, or the above.', prompt_text)
        self.assertIn('For lookup or search requests, explicitly state what should be searched for so downstream steps do not need the original conversation to know the target.', prompt_text)
        self.assertIn('full conversation context will not be passed through later', prompt_text)
        self.assertIn('one of the provided available agent names', prompt_text)
        self.assertIn('"agent": "main_agent"', prompt_text)
        self.assertIn('"agent": "profile_management"', prompt_text)

    def test_loaded_user_agent_is_exposed_to_request_analysis(self) -> None:
        user_profile = UserProfile(user_id="test-user")
        custom_agent = UserAgent.model_validate(
            {
                "id": str(uuid4()),
                "user_id": "test-user",
                "name": "trip_planner",
                "description": "Specialized travel planning agent.",
                "allowed_categories": ["web_search", "calendar"],
                "planner_instruction": "You are a travel planner.",
                "planner_rules": "",
                "max_turns": 10,
                "is_active": True,
                "metadata": {},
            }
        )
        state = MainState.new(
            task="Plan me a trip to Tokyo.",
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=user_profile,
            ),
            llm=MockLLM([]),
            agent_profiles=_agent_profiles_for(user_profile),
        )

        with patch(
            "request_orchestrator.shared.agents.load_user_agents.get_user_agent_repo",
            return_value=FakeUserAgentRepository([custom_agent]),
        ):
            from request_orchestrator.shared.agents import load_user_agents

            load_user_agents(state)

        prompt_text = build_request_analysis_prompt(state).build()

        self.assertIn('"agent": "trip_planner"', prompt_text)
        self.assertIn('"description": "Specialized travel planning agent."', prompt_text)

    def test_main_state_gathers_child_tool_results_for_synthesis_evidence(self) -> None:
        main_state = MainState.new(
            task='Combine child agent evidence.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profiles=_agent_profiles_for(UserProfile()),
        )

        profile_state = main_state.agent_states['profile_management']
        profile_state.result = AgentResult(
            tool_results=[
                ToolResult(
                    step_id=namespace_step_id('profile_management', 'P1E1'),
                    tool_name='get_current_weather',
                    iteration=1,
                    result={'temperature': 21.2},
                    evidence_views=[
                        EvidenceView(
                            evidence_id='profile_management:P1E1R1',
                            item_id='Toronto',
                            title='Weather Result',
                            summary='21.2 C in Toronto.',
                        )
                    ],
                    hydrated_evidence=[
                        HydratedEvidence(
                            evidence_id='profile_management:P1E1R1',
                            item_id='Toronto',
                            title='Weather Result',
                            summary='21.2 C in Toronto.',
                            source='get_current_weather',
                            entity_type='weather',
                        )
                    ],
                )
            ]
        )

        main_agent_state = main_state.agent_states['main_agent']
        main_agent_state.result = AgentResult(
            tool_results=[
                ToolResult(
                    step_id=namespace_step_id('main_agent', 'P1E1'),
                    tool_name='generic_web_search',
                    iteration=1,
                    result={'items': ['ramen']},
                    evidence_views=[
                        EvidenceView(
                            evidence_id='main_agent:P1E1R1',
                            item_id='ramen-1',
                            title='Ramen Result',
                            summary='Popular ramen shop.',
                        )
                    ],
                    hydrated_evidence=[
                        HydratedEvidence(
                            evidence_id='main_agent:P1E1R1',
                            item_id='ramen-1',
                            title='Ramen Result',
                            summary='Popular ramen shop.',
                            source='generic_web_search',
                            entity_type='web_search_results',
                        )
                    ],
                )
            ],
            relevant_evidence_ids=['main_agent:P1E1R1'],
        )

        tool_results = main_state.gather_tool_results()
        evidence_bundle = build_evidence_bundle_from_tool_results(tool_results)
        evidence_steps = build_evidence_steps_from_tool_results(
            tool_results,
            evidence_bundle.evidence_views_by_step_id,
        )

        self.assertEqual(len(tool_results), 2)
        self.assertEqual(
            [tool_result.evidence_views[0].title for tool_result in tool_results],
            ['Weather Result', 'Ramen Result'],
        )
        self.assertEqual(
            sorted(evidence_bundle.hydrated_evidence_by_id.keys()),
            ['main_agent:P1E1R1', 'profile_management:P1E1R1'],
        )
        self.assertEqual(main_state.gather_relevant_evidence_ids(), ['main_agent:P1E1R1'])
        self.assertEqual(len(evidence_steps), 2)
        self.assertEqual(
            [step.evidence[0].title for step in evidence_steps],
            ['Weather Result', 'Ramen Result'],
        )

    def test_main_state_distributes_goals_to_matching_child_agents(self) -> None:
        main_state = MainState.new(
            task='Handle a routed request.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profiles=_agent_profiles_for(UserProfile()),
        )
        main_state.request_analysis = RequestAnalysis(
            goals=[
                RequestAnalysisGoal(
                    agent='main_agent',
                    goal='Find current transit details for Toronto.',
                    tool_categories=['calendar', 'web_search'],
                ),
                RequestAnalysisGoal(
                    agent='future_agent',
                    goal='Perform a future specialized task.',
                    tool_categories=['knowledge'],
                ),
            ],
            requested_user_attribute_types=['projects.goals'],
        )

        main_state.initialize_agent_states()

        profile_state = main_state.agent_states['profile_management']
        routed_main_state = main_state.agent_states['main_agent']

        self.assertEqual(profile_state.inputs.task, '')
        self.assertEqual(profile_state.inputs.tool_category_names, [])
        self.assertEqual(routed_main_state.inputs.task, 'Find current transit details for Toronto.')
        self.assertEqual(routed_main_state.inputs.tool_category_names, ['calendar', 'web_search'])
        self.assertEqual(len(main_state.request_analysis.goals), 2)

    def test_build_llm_for_stage_reuses_existing_non_chat_llm(self) -> None:
        parent_state = AgentState.new(
            task='Please remember that I like pizza.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        self.assertIs(
            build_llm_for_stage(
                execution_context=parent_state.execution_context,
                llm=parent_state.llm,
                agent=parent_state.resolve_agent_scope(),
                stage='planner',
                reuse_llm_for_agent_scope=parent_state.resolve_agent_scope(),
            ),
            parent_state.llm,
        )

    def test_prompt_text_prunes_null_fields_from_plan_evidence(self) -> None:
        prompt = AgentPrompt(
            instruction='Synthesize the answer.',
            user_profile=UserProfile(),
            evidence=[
                EvidenceStep(
                    type='generic',
                    evidence=[
                        EvidenceView(
                            evidence_id="P1E1R1",
                            item_id='https://example.com/soba',
                            title='Soba Noodles',
                            summary='Authentic soba noodles',
                        )
                    ],
                )
            ],
            schema='{}',
        )
        prompt.include_section(PromptSectionKeys.USER_PROFILE)
        prompt.include_section(PromptSectionKeys.EVIDENCE)
        prompt.include_section(PromptSectionKeys.SCHEMA)

        prompt_text = prompt.build()

        self.assertIn('"title": "Soba Noodles"', prompt_text)
        self.assertIn('"evidence_id": "P1E1R1"', prompt_text)
        self.assertIn('"item_id": "https://example.com/soba"', prompt_text)
        self.assertIn('"summary": "Authentic soba noodles"', prompt_text)
        self.assertNotIn('image_url', prompt_text)
        self.assertNotIn('category', prompt_text)

    def test_synthesis_prompt_excludes_recent_roundtrips(self) -> None:
        state = AgentState.new(
            task='Summarize this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(
                    conversation_summary='User has been comparing pantry noodle options and cares about quick preparation.',
                    latest_conversation_summary='User was comparing noodle options.',
                    tool_summary='Earlier tools found soba, udon, and ramen options.',
                    recent_roundtrips=[
                        RecentRoundtrip(
                            message_index=4,
                            user_prompt='Earlier user prompt',
                            roundtrip_summary='Earlier roundtrip summary',
                        )
                    ],
                ),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        prompt = build_synthesis_prompt(
            evidence=[
                EvidenceStep(
                    type='web_search_results',
                    evidence=[
                        EvidenceView(
                            evidence_id="P1E1R1",
                            item_id="result-1",
                            title='Result',
                            summary='Evidence result.',
                        )
                    ],
                )
            ],
            state=state,
        )
        prompt_text = prompt.build()

        self.assertIn('"conversation_context": {', prompt_text)
        self.assertIn('conversation_summary', prompt_text)
        self.assertIn('User has been comparing pantry noodle options and cares about quick preparation.', prompt_text)
        self.assertIn('latest_conversation_summary', prompt_text)
        self.assertIn('tool_summary', prompt_text)
        self.assertIn('Earlier tools found soba, udon, and ramen options.', prompt_text)
        self.assertNotIn('recent_roundtrips', prompt_text)
        self.assertNotIn('Earlier user prompt', prompt_text)
        self.assertNotIn('Earlier roundtrip summary', prompt_text)

    def test_synthesis_prompt_includes_evidence_object_when_present(self) -> None:
        state = AgentState.new(
            task='Summarize this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        prompt = build_synthesis_prompt(
            evidence=[
                EvidenceStep(
                    type='decks',
                    evidence=[
                        EvidenceView(
                            evidence_id="P1E1R1",
                            item_id="uril-the-miststalker",
                            title='Uril, the Miststalker (Commander)',
                            summary='EDHREC summary.',
                            evidence_object={
                                "top_themes": [
                                    {"value": "Auras", "count": 1218},
                                ]
                            },
                        )
                    ],
                )
            ],
            state=state,
        )

        prompt_text = prompt.build()

        self.assertIn('"evidence_object"', prompt_text)
        self.assertIn('"top_themes"', prompt_text)
        self.assertIn('"Auras"', prompt_text)

    def test_synthesis_prompt_omits_empty_conversation_context_section(self) -> None:
        state = AgentState.new(
            task='Summarize this.',
            execution_context=AgentExecutionContext.new(
                conversation_context=ConversationContext(),
                user_profile=UserProfile(),
            ),
            llm=MockLLM([]),
            agent_profile=MAIN_AGENT_PROFILE,
        )

        prompt = build_synthesis_prompt(
            evidence=[
                EvidenceStep(
                    type='web_search_results',
                    evidence=[
                        EvidenceView(
                            evidence_id="P1E1R1",
                            item_id="result-1",
                            title='Result',
                            summary='Evidence result.',
                        )
                    ],
                )
            ],
            state=state,
        )

        prompt_text = prompt.build()

        self.assertNotIn('conversation_context:', prompt_text)
        self.assertNotIn('conversation_context: {', prompt_text)
        self.assertNotIn('\n\n{}\n\n', prompt_text)

    def test_user_attribute_creation_orchestration(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goals": [
            {
              "agent": "profile_management",
              "goal": "Store the user's stated food preferences as durable profile data.",
              "tool_categories": ["user_attributes"]
            },
            {
              "agent": "main_agent",
              "goal": "Store the user's food preferences.",
              "tool_categories": ["user_attributes"]
            }
          ],
          "requested_user_attribute_types": []
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
          ]}
        """

        profile_management_completion_response = """
        {
          "steps": []}
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
          ]}
        """

        synthesis_response = """
        {
          "result": [{"content": "Stored your food likes as a user attribute.", "evidence_ids": []}],
          "next_question": "Do you want me to remember any other food preferences?",
          "roundtrip_summary": "Stored the user's stated food likes as a persistent user attribute and confirmed the profile state.",
          "tool_summary": {
            "produced": ["current user attribute list"],
            "entities": ["pizza", "eggs"]
          }
        }
        """

        result, llm, _, _ = self._run_case(
            user_query='Please remember that I like pizza and eggs.',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[
                    profile_management_planner_response,
                    profile_management_completion_response],
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
                )],
        )

        self.assertEqual(result.answer, ['Stored your food likes as a user attribute.'])
        self.assertGreaterEqual(len(llm.invocations), 2)
        self.assertIn('"task": "Please remember that I like pizza and eggs."', llm.prompts[0] or '')
        self.assertNotIn('"latest_user_prompt":', llm.prompts[0] or '')
        self.assertNotIn('"user_profile":', llm.prompts[0] or '')
        self.assertTrue(any('conversation_context:' not in (prompt or '') for prompt in llm.prompts))
        self.assertNotIn('recent_roundtrip_tool_summaries', llm.prompts[-1] or '')
        self.assertNotIn('conversation_summary', llm.prompts[-1] or '')
        self.assertLessEqual((llm.prompts[-1] or '').count('message_index'), 3)

    def test_requested_user_attributes_are_loaded_after_request_analysis(self) -> None:
        fake_repo = FakeUserAttributeRepository(
            created_attributes=[
                UserAttribute(
                    id=uuid4(),
                    user_id='test-user',
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
                    user_id='test-user',
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
          "goals": [
            {
              "agent": "main_agent",
              "goal": "Use the user's food preferences to help with the request.",
              "tool_categories": ["food"]
            }
          ],
          "requested_user_attribute_types": ["food.likes"]
        }
        """

        profile_management_planner_response = """
        {
          "steps": []}
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
          ]}
        """

        synthesis_response = """
        {
          "result": [{"content": "I used your stored food preferences while looking up a meal idea.", "evidence_ids": []}],
          "next_question": "Do you want a few more options based on those preferences?",
          "roundtrip_summary": "Loaded the user's stored food likes and used them while planning a meal-related response.",
          "tool_summary": {
            "produced": ["meal ideas"],
            "entities": ["pizza"]
          }
        }
        """

        result, llm, repo, roundtrip_id = self._run_case(
            user_query='Use what you know about my food preferences to suggest something.',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[],
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
                )],
        )

        self.assertEqual(result.answer, ['I used your stored food preferences while looking up a meal idea.'])
        self.assertGreaterEqual(len(llm.invocations), 2)
        self.assertNotIn('"user_profile":', llm.prompts[0] or '')

        with patch('common.logging.conversation_event_view.get_conversation_repo', return_value=repo):
            fetched_logs = fetch_agent_logs_for_roundtrip(roundtrip_id)

        main_agent_logs = fetched_logs.get('main_agent', [])
        orchestrator_logs = fetched_logs.get('request_orchestrator', [])
        request_analysis_log = next(log for log in orchestrator_logs if log.get('kind') == 'request_analysis')
        profile_load_log = next(log for log in orchestrator_logs if log.get('kind') == 'profile_load')
        synthesis_log = next(log for log in orchestrator_logs if log.get('kind') == 'synthesis')

        self.assertIsInstance(request_analysis_log.get('data', {}).get('requested_user_attribute_types'), list)
        self.assertIsInstance(profile_load_log.get('data', {}).get('loaded_attribute_count'), int)
        self.assertIsInstance(profile_load_log.get('data', {}).get('loaded_attribute_types'), list)
        self.assertIsInstance(profile_load_log.get('data', {}).get('loaded_attributes'), list)
        self.assertEqual(
            synthesis_log.get('data', {}).get('answer_preview'),
            ['I used your stored food preferences while looking up a meal idea.'],
        )

    def test_calculate_tool_orchestration(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goals": [
            {
              "agent": "main_agent",
              "goal": "Calculate the result of the math expression.",
              "tool_categories": ["math"]
            }
          ],
          "requested_user_attribute_types": []
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
          ]}
        """

        profile_management_completion_response = """
        {
          "steps": []}
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
          ]}
        """

        synthesis_response = """
        {
          "result": [{"content": "The result is 47.0.", "evidence_ids": []}],
          "next_question": "Do you want me to show the calculation steps too?",
          "roundtrip_summary": "Calculated the requested expression using the math tool and returned the numeric result.",
          "tool_summary": {
            "produced": ["numeric result"],
            "entities": ["(15 * 8) / 3 + 7"]
          }
        }
        """

        result, llm, _, _ = self._run_case(
            user_query='What is (15 * 8) / 3 + 7?',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[],
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
        self.assertGreaterEqual(len(llm.invocations), 2)

    def test_world_time_tool_orchestration(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goals": [
            {
              "agent": "main_agent",
              "goal": "Find the current time in Tokyo.",
              "tool_categories": ["calendar"]
            }
          ],
          "requested_user_attribute_types": []
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
          ]}
        """

        profile_management_completion_response = """
        {
          "steps": []}
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
          ]}
        """

        synthesis_response = """
        {
          "result": [{"content": "The current time in Tokyo is 2026-08-04T21:30:00+09:00.", "evidence_ids": []}],
          "next_question": "Do you want the current date there as well?",
          "roundtrip_summary": "Looked up the current time in Tokyo using the world time tool and returned the reported local datetime.",
          "tool_summary": {
            "produced": ["local datetime", "UTC offset"],
            "entities": ["Asia/Tokyo"]
          }
        }
        """

        result, llm, _, _ = self._run_case(
            user_query='What time is it in Tokyo right now?',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[],
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
        self.assertGreaterEqual(len(llm.invocations), 2)

    def test_next_question_is_preserved(self) -> None:
        fake_repo = FakeUserAttributeRepository()

        request_analysis_response = """
        {
          "goals": [
            {
              "agent": "main_agent",
              "goal": "Clarify an underspecified request.",
              "tool_categories": []
            }
          ],
          "requested_user_attribute_types": []
        }
        """

        profile_management_planner_response = """
        {
          "steps": []}
        """

        main_planner_response = """
        {
          "steps": []}
        """

        synthesis_response = """
        {
          "result": [{"content": "I can help with that.", "evidence_ids": []}],
          "next_question": "Which product category do you want to focus on?",
          "roundtrip_summary": "The request was underspecified, so the response preserved the clarifying question and dropped the follow-up variant.",
          "tool_summary": {
            "produced": [],
            "entities": []
          }
        }
        """

        result, _, _, _ = self._run_case(
            user_query='Help me pick something.',
            llm_responses=MockLLMScenario(
                request_analysis=request_analysis_response,
                profile_planner=[],
                main_planner=[main_planner_response],
                synthesis=[synthesis_response],
            ),
            patchers=[
                patch(
                    'personalization.profile.service.get_user_attribute_repo',
                    return_value=fake_repo,
                )],
        )

        self.assertEqual(result.next_question, 'Which product category do you want to focus on?')


if __name__ == '__main__':
    unittest.main()







