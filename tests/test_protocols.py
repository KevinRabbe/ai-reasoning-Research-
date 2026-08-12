from ai_reasoning.protocols import LearnedProtocolSpec, StructuredProtocol, TextProtocol
from ai_reasoning.tasks import generate_constraint_task, generate_graph_task


def test_text_protocol_bit_cost_is_utf8_exact():
    task = generate_constraint_task(4, entity_count=6)
    message = TextProtocol().encode(task)
    assert message.bit_cost == len(message.payload) * 8
    assert message.symbol_count == len(message.payload)


def test_structured_protocol_is_more_compact_on_reference_tasks():
    protocols = (TextProtocol(), StructuredProtocol())
    for task in (generate_constraint_task(7, 6), generate_graph_task(7, 8)):
        text, structured = (protocol.encode(task) for protocol in protocols)
        assert structured.bit_cost < text.bit_cost


def test_learned_channel_contract_is_frozen():
    spec = LearnedProtocolSpec()
    assert spec.vocabulary_size == 256
    assert spec.bits_per_symbol == 8
    assert spec.max_symbols == 64
    assert spec.stop_symbol == 0
