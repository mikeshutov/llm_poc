from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

from llm.conversation_model_config import CONVERSATION_MODEL_CONFIG_SPECS, OPENAI_PROVIDER
from llm.repository.conversation_model_config_repository import ConversationModelConfigRepository
from conversation.constants import CONVERSATION_SOURCE_STREAMLIT
from conversation.repository.conversation_repository import ConversationRepository
from conversation.models.conversation_models import ConversationMetadata


class FakeCursor:
    def __init__(self, fetchone_row=None, fetchall_rows=None):
        self.fetchone_row = fetchone_row
        self.fetchall_rows = fetchall_rows or []
        self.executed: list[tuple[str, tuple]] = []
        self.rowcount = 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params):
        self.executed.append((sql, params))

    def fetchone(self):
        return self.fetchone_row

    def fetchall(self):
        return self.fetchall_rows


class FakeConnection:
    def __init__(self, cursors):
        self._cursors = list(cursors)

    def cursor(self, row_factory=None):
        return self._cursors.pop(0)


def test_create_llm_call_round_trips_decimal_and_nullable_ids() -> None:
    row = {
        'id': uuid4(),
        'conversation_id': None,
        'roundtrip_id': None,
        'agent': 'utility',
        'stage': 'image_caption',
        'callsite': 'llm_client.generate_caption_from_image_file',
        'model': 'gpt-5.6-luna',
        'input_tokens': 50,
        'output_tokens': 10,
        'total_tokens': 60,
        'cached_input_tokens': 0,
        'input_price_per_million_tokens': Decimal('1.00'),
        'output_price_per_million_tokens': Decimal('6.00'),
        'computed_input_cost': Decimal('0.00005'),
        'computed_output_cost': Decimal('0.00006'),
        'computed_total_cost': Decimal('0.00011'),
        'metadata': {'kind': 'caption'},
        'created_at': '2026-08-09T00:00:00Z',
        'updated_at': '2026-08-09T00:00:00Z',
    }
    cursor = FakeCursor(fetchone_row=row)

    with patch('conversation.repository.conversation_repository.register_vector', lambda conn: None):
        repo = ConversationRepository(conn=FakeConnection([cursor]))

    record = repo.create_llm_call(
        conversation_id=None,
        roundtrip_id=None,
        user_id=None,
        agent='utility',
        stage='image_caption',
        callsite='llm_client.generate_caption_from_image_file',
        model='gpt-5.6-luna',
        input_tokens=50,
        output_tokens=10,
        total_tokens=60,
        cached_input_tokens=0,
        input_price_per_million_tokens=Decimal('1.00'),
        output_price_per_million_tokens=Decimal('6.00'),
        computed_input_cost=Decimal('0.00005'),
        computed_output_cost=Decimal('0.00006'),
        computed_total_cost=Decimal('0.00011'),
        metadata={'kind': 'caption'},
    )

    assert 'INSERT INTO llm_call' in cursor.executed[0][0]
    assert record.conversation_id is None
    assert record.roundtrip_id is None
    assert record.computed_total_cost == Decimal('0.00011')


def test_list_llm_calls_for_roundtrip_returns_rows_in_order() -> None:
    roundtrip_id = uuid4()
    rows = [
        {
            'id': uuid4(),
            'conversation_id': uuid4(),
            'roundtrip_id': roundtrip_id,
            'agent': 'main_agent',
            'stage': 'request_analysis',
            'callsite': 'request_analysis.analyze_request',
            'model': 'gpt-5.6-luna',
            'input_tokens': 100,
            'output_tokens': 20,
            'total_tokens': 120,
            'cached_input_tokens': 0,
        'input_price_per_million_tokens': Decimal('1.00'),
            'output_price_per_million_tokens': Decimal('6.00'),
            'computed_input_cost': Decimal('0.0001'),
            'computed_output_cost': Decimal('0.00012'),
            'computed_total_cost': Decimal('0.00022'),
            'metadata': {},
            'created_at': '2026-08-09T00:00:00Z',
            'updated_at': '2026-08-09T00:00:00Z',
        },
        {
            'id': uuid4(),
            'conversation_id': uuid4(),
            'roundtrip_id': roundtrip_id,
            'agent': 'main_agent',
            'stage': 'synthesis',
            'callsite': 'shared_synthesis.run_synthesis',
            'model': 'gpt-5.6-terra',
            'input_tokens': 140,
            'output_tokens': 50,
            'total_tokens': 190,
            'cached_input_tokens': 0,
            'input_price_per_million_tokens': Decimal('2.50'),
            'output_price_per_million_tokens': Decimal('15.00'),
            'computed_input_cost': Decimal('0.00035'),
            'computed_output_cost': Decimal('0.00075'),
            'computed_total_cost': Decimal('0.0011'),
            'metadata': {},
            'created_at': '2026-08-09T00:01:00Z',
            'updated_at': '2026-08-09T00:01:00Z',
        },
    ]
    cursor = FakeCursor(fetchall_rows=rows)

    with patch('conversation.repository.conversation_repository.register_vector', lambda conn: None):
        repo = ConversationRepository(conn=FakeConnection([cursor]))

    records = repo.list_llm_calls_for_roundtrip(roundtrip_id)

    assert 'FROM llm_call' in cursor.executed[0][0]
    assert [record.stage for record in records] == ['request_analysis', 'synthesis']


def test_create_conversation_persists_default_model_config_rows() -> None:
    row = {
        'id': uuid4(),
        'user_id': 'anonymous',
        'title': 'anonymous',
        'created_at': '2026-08-09T00:00:00Z',
        'metadata': {'sources': [CONVERSATION_SOURCE_STREAMLIT]},
        'tone_state': {},
        'summary': '',
    }
    create_cursor = FakeCursor(fetchone_row=row)
    config_cursors = [
        FakeCursor(fetchone_row={
            'conversation_id': row['id'],
            'agent': spec.agent,
            'stage': spec.stage,
            'provider': OPENAI_PROVIDER,
            'model': 'gpt-5.6-luna' if spec.stage in {'request_analysis', 'reranker', 'evaluator'} or spec.agent == 'profile_agent' else 'gpt-5.6-terra',
            'created_at': '2026-08-09T00:00:00Z',
            'updated_at': '2026-08-09T00:00:00Z',
        })
        for spec in CONVERSATION_MODEL_CONFIG_SPECS
    ]

    with patch('conversation.repository.conversation_repository.register_vector', lambda conn: None):
        repo = ConversationRepository(conn=FakeConnection([create_cursor, FakeCursor(fetchall_rows=[]), *config_cursors]))

    conversation = repo.create_conversation(
        user_id='anonymous',
        metadata=ConversationMetadata(sources=[CONVERSATION_SOURCE_STREAMLIT]),
    )

    assert conversation.id == row['id']
    assert 'INSERT INTO conversation' in create_cursor.executed[0][0]
    assert len(config_cursors) == len(CONVERSATION_MODEL_CONFIG_SPECS)
    assert all('INSERT INTO conversation_model_config' in cursor.executed[0][0] for cursor in config_cursors)



def test_resolve_conversation_model_config_backfills_missing_default_rows() -> None:
    conversation_id = uuid4()
    existing_rows = [
        {
            'conversation_id': conversation_id,
            'agent': 'main_agent',
            'stage': 'planner',
            'provider': OPENAI_PROVIDER,
            'model': 'gpt-5.6-terra',
            'created_at': '2026-08-09T00:00:00Z',
            'updated_at': '2026-08-09T00:00:00Z',
        }
    ]
    list_cursor = FakeCursor(fetchall_rows=existing_rows)
    backfill_cursors = [
        FakeCursor(fetchone_row={
            'conversation_id': conversation_id,
            'agent': spec.agent,
            'stage': spec.stage,
            'provider': OPENAI_PROVIDER,
            'model': 'gpt-5.6-luna' if spec.stage in {'request_analysis', 'reranker', 'evaluator'} or spec.agent == 'profile_agent' else 'gpt-5.6-terra',
            'created_at': '2026-08-09T00:00:00Z',
            'updated_at': '2026-08-09T00:00:00Z',
        })
        for spec in CONVERSATION_MODEL_CONFIG_SPECS
        if not (spec.agent == 'main_agent' and spec.stage == 'planner')
    ]

    repo = ConversationModelConfigRepository(conn=FakeConnection([list_cursor, *backfill_cursors]))

    resolved = repo.resolve(conversation_id)

    assert resolved.main_agent.planner.model == 'gpt-5.6-terra'
    assert resolved.main_agent.request_analysis.model == 'gpt-5.6-luna'
    assert resolved.profile_agent.planner.model == 'gpt-5.6-luna'
    assert resolved.shared.evaluator.model == 'gpt-5.6-luna'
    assert resolved.shared.reranker.model == 'gpt-5.6-luna'
    assert len(backfill_cursors) == len(CONVERSATION_MODEL_CONFIG_SPECS) - 1
    assert all('INSERT INTO conversation_model_config' in cursor.executed[0][0] for cursor in backfill_cursors)
