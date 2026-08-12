import torch

from ai_reasoning.model import ByteReasoningSystem, PAD_BYTE, batch_byte_messages
from ai_reasoning.protocols import ProtocolMessage, StructuredProtocol, TextProtocol
from ai_reasoning.tasks import QueryType, Relation, RelationType, Task, TaskFamily


def _task() -> Task:
    return Task(
        family=TaskFamily.GRAPH,
        entity_count=4,
        relations=(
            Relation(RelationType.EDGE, 0, 1),
            Relation(RelationType.EDGE, 1, 3),
            Relation(RelationType.EDGE, 0, 2),
            Relation(RelationType.EDGE, 2, 3),
        ),
        query_type=QueryType.NEXT_STEP,
        seed=7,
        start=0,
        goal=3,
    )


def _tiny_system() -> ByteReasoningSystem:
    return ByteReasoningSystem(
        d_model=32,
        encoder_heads=4,
        encoder_layers=1,
        controller_heads=4,
        controller_depth=1,
        num_slots=4,
        ff_dim=64,
        dropout=0.0,
        max_symbols=1024,
    )


def test_batching_preserves_protocol_accounting_and_padding() -> None:
    messages = [
        ProtocolMessage(payload=b"abc", bit_cost=24, symbol_count=3),
        ProtocolMessage(payload=b"x", bit_cost=8, symbol_count=1),
    ]
    batch = batch_byte_messages(messages)
    assert batch.tokens.tolist() == [[97, 98, 99], [120, PAD_BYTE, PAD_BYTE]]
    assert batch.padding_mask.tolist() == [[False, False, False], [False, True, True]]
    assert batch.bit_costs.tolist() == [24, 8]
    assert batch.symbol_counts.tolist() == [3, 1]


def test_text_and_ir_use_the_same_neural_input_contract() -> None:
    task = _task()
    text = TextProtocol().encode(task)
    structured = StructuredProtocol().encode(task)
    batch = batch_byte_messages([text, structured])
    assert batch.batch_size == 2
    assert batch.max_symbols == max(text.symbol_count, structured.symbol_count)
    assert batch.bit_costs.tolist() == [text.bit_cost, structured.bit_cost]


def test_shared_system_forward_shape_and_intermediate_states() -> None:
    torch.manual_seed(11)
    task = _task()
    messages = [TextProtocol().encode(task), StructuredProtocol().encode(task)]
    batch = batch_byte_messages(messages)
    model = _tiny_system().eval()
    output = model(batch, reasoning_steps=3, return_intermediate=True)
    assert output.workspace.shape == (2, 4, 32)
    assert len(output.intermediate_states) == 3
    assert all(state.shape == (2, 4, 32) for state in output.intermediate_states)


def test_reasoning_steps_reuse_parameters_instead_of_unrolling_modules() -> None:
    model = _tiny_system()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert len(model.controller.blocks) == 1

    batch = batch_byte_messages([TextProtocol().encode(_task())])
    model.eval()
    with torch.no_grad():
        model(batch, reasoning_steps=1)
        model(batch, reasoning_steps=9)

    assert sum(parameter.numel() for parameter in model.parameters()) == parameter_count
    assert len(model.controller.blocks) == 1


def test_eval_forward_is_deterministic() -> None:
    torch.manual_seed(23)
    model = _tiny_system().eval()
    batch = batch_byte_messages([TextProtocol().encode(_task())])
    with torch.no_grad():
        first = model(batch, reasoning_steps=4).workspace
        second = model(batch, reasoning_steps=4).workspace
    assert torch.equal(first, second)


def test_gradients_reach_byte_embedding_and_workspace() -> None:
    torch.manual_seed(31)
    model = _tiny_system().train()
    batch = batch_byte_messages([StructuredProtocol().encode(_task())])
    output = model(batch, reasoning_steps=2)
    output.workspace.square().mean().backward()
    assert model.message_encoder.embedding.weight.grad is not None
    assert model.controller.initial_workspace.grad is not None
    assert torch.isfinite(model.message_encoder.embedding.weight.grad).all()
    assert torch.isfinite(model.controller.initial_workspace.grad).all()


def test_system_rejects_messages_above_encoder_limit() -> None:
    model = ByteReasoningSystem(
        d_model=32,
        encoder_heads=4,
        encoder_layers=1,
        controller_heads=4,
        controller_depth=1,
        num_slots=4,
        ff_dim=64,
        dropout=0.0,
        max_symbols=4,
    )
    batch = batch_byte_messages([ProtocolMessage(payload=b"12345", bit_cost=40, symbol_count=5)])
    try:
        model(batch)
    except ValueError as error:
        assert "encoder limit" in str(error)
    else:
        raise AssertionError("expected message length validation")
