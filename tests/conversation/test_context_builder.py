from __future__ import annotations

from uuid import uuid4
from unittest.mock import patch

from conversation.context_builder import build_roundtrip_context
from conversation.models.conversation_models import Conversation, ConversationRoundtrip, ConversationSummary


class FakeConversationRepository:
    def __init__(self) -> None:
        self.list_roundtrips_calls: list[dict[str, object]] = []

    def get_conversation(self, conversation_id):
        return Conversation(
            id=conversation_id,
            user_id='user-1',
            title='Test',
            created_at='2026-08-05T00:00:00Z',
            metadata={},
            tone_state={},
            summary='top level summary',
            summary_embedding=None,
        )

    def get_latest_summary(self, conversation_id):
        return ConversationSummary(
            id=uuid4(),
            conversation_id=conversation_id,
            summary='latest batch summary',
            tool_summary='latest tool summary',
            message_index_cutoff=5,
            created_at='2026-08-05T00:00:00Z',
        )

    def list_roundtrips(self, conversation_id, limit=50, after_message_index=None, newest_first=False):
        self.list_roundtrips_calls.append(
            {
                'conversation_id': conversation_id,
                'limit': limit,
                'after_message_index': after_message_index,
                'newest_first': newest_first,
            }
        )
        return [
            ConversationRoundtrip(
                id=uuid4(),
                conversation_id=conversation_id,
                message_index=6,
                user_prompt='user six',
                generated_response='assistant six',
                roundtrip_summary='summary six',
                roundtrip_summary_embedding=None,
                response_payload={},
                parsed_query={},
                created_at='2026-08-05T00:00:00Z',
                metadata={},
                model=None,
                feedback_id=None,
            ),
            ConversationRoundtrip(
                id=uuid4(),
                conversation_id=conversation_id,
                message_index=7,
                user_prompt='user seven',
                generated_response='assistant seven',
                roundtrip_summary='summary seven',
                roundtrip_summary_embedding=None,
                response_payload={},
                parsed_query={},
                created_at='2026-08-05T00:00:00Z',
                metadata={},
                model=None,
                feedback_id=None,
            ),
            ConversationRoundtrip(
                id=uuid4(),
                conversation_id=conversation_id,
                message_index=8,
                user_prompt='user eight',
                generated_response='assistant eight',
                roundtrip_summary='summary eight',
                roundtrip_summary_embedding=None,
                response_payload={},
                parsed_query={},
                created_at='2026-08-05T00:00:00Z',
                metadata={},
                model=None,
                feedback_id=None,
                assistant_follow_up='Would you like more options?',
            ),
        ]

    def get_latest_completed_roundtrip(self, conversation_id):
        return ConversationRoundtrip(
            id=uuid4(),
            conversation_id=conversation_id,
            message_index=8,
            user_prompt='user eight',
            generated_response='assistant eight',
            roundtrip_summary='summary eight',
            roundtrip_summary_embedding=None,
            response_payload={},
            parsed_query={},
            created_at='2026-08-05T00:00:00Z',
            metadata={},
            model=None,
            assistant_follow_up='Would you like more options?',
        )


def test_build_roundtrip_context_requests_latest_unsummarized_roundtrips() -> None:
    fake_repo = FakeConversationRepository()
    conversation_id = str(uuid4())

    with patch('conversation.context_builder.get_conversation_repo', return_value=fake_repo):
        context = build_roundtrip_context(conversation_id, limit=3)

    assert fake_repo.list_roundtrips_calls == [
        {
            'conversation_id': fake_repo.list_roundtrips_calls[0]['conversation_id'],
            'limit': 3,
            'after_message_index': 5,
            'newest_first': True,
        }
    ]
    assert [roundtrip.message_index for roundtrip in context.recent_roundtrips] == [6, 7, 8]
    assert [roundtrip.user_prompt for roundtrip in context.recent_roundtrips] == ['user six', 'user seven', 'user eight']
    assert context.recent_roundtrips[-1].assistant_follow_up == 'Would you like more options?'
    assert context.previous_user_request == 'user eight'
    assert context.latest_assistant_follow_up == 'Would you like more options?'


def test_build_roundtrip_context_uses_completed_roundtrips_from_repository() -> None:
    fake_repo = FakeConversationRepository()
    conversation_id = str(uuid4())

    def list_completed_roundtrips(conversation_id, limit=50, after_message_index=None, newest_first=False):
        fake_repo.list_roundtrips_calls.append(
            {
                'conversation_id': conversation_id,
                'limit': limit,
                'after_message_index': after_message_index,
                'newest_first': newest_first,
            }
        )
        return [
            ConversationRoundtrip(
                id=uuid4(),
                conversation_id=conversation_id,
                message_index=6,
                user_prompt='user six',
                generated_response='assistant six',
                roundtrip_summary='summary six',
                roundtrip_summary_embedding=None,
                response_payload={},
                parsed_query={},
                created_at='2026-08-05T00:00:00Z',
                    metadata={},
                    model=None,
                    feedback_id=None,
                ),
            ConversationRoundtrip(
                id=uuid4(),
                conversation_id=conversation_id,
                message_index=8,
                user_prompt='user eight',
                generated_response='assistant eight',
                roundtrip_summary='summary eight',
                roundtrip_summary_embedding=None,
                response_payload={},
                parsed_query={},
                created_at='2026-08-05T00:00:00Z',
                metadata={},
                model=None,
                feedback_id=None,
            ),
        ]

    fake_repo.list_roundtrips = list_completed_roundtrips

    with patch('conversation.context_builder.get_conversation_repo', return_value=fake_repo):
        context = build_roundtrip_context(conversation_id, limit=3)

    assert [roundtrip.message_index for roundtrip in context.recent_roundtrips] == [6, 8]
    assert [roundtrip.user_prompt for roundtrip in context.recent_roundtrips] == ['user six', 'user eight']



def test_build_conversation_context_json_prunes_empty_fields() -> None:
    from conversation.models.conversation_models import ConversationContext, RecentRoundtrip
    from conversation.utils import build_conversation_context_json

    context = ConversationContext(
        conversation_summary='',
        latest_conversation_summary='',
        tool_summary='',
        recent_roundtrips=[
            RecentRoundtrip(
                message_index=1,
                user_prompt='hello',
                roundtrip_summary='',
                assistant_follow_up='Would you like another option?',
            )
        ],
        recent_roundtrip_tool_summaries=[],
    )

    rendered = build_conversation_context_json(context)

    assert 'latest_conversation_summary' not in rendered
    assert 'tool_summary' not in rendered
    assert 'roundtrip_summary' not in rendered
    assert 'hello' in rendered
    assert 'Would you like another option?' in rendered
