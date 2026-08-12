import torch

from ai_reasoning.model import GraphBaselineModel
from ai_reasoning.protocols import StructuredProtocol, TextProtocol
from ai_reasoning.training.graph import GraphTrainConfig, graph_batch, train_graph_baseline


def _tiny_kwargs():
    return dict(d_model=32, encoder_heads=4, encoder_layers=1, controller_heads=4, controller_depth=1, num_slots=4, ff_dim=64, dropout=0.0, max_symbols=2048)


def test_graph_baseline_logits_match_fixed_iid_entity_count() -> None:
    batch, _ = graph_batch(protocol=StructuredProtocol(), seeds=[1, 2], entity_count=8, device="cpu")
    model = GraphBaselineModel(entity_count=8, **_tiny_kwargs()).eval()
    with torch.no_grad():
        logits = model(batch, reasoning_steps=2)
    assert logits.shape == (2, 8)


def test_text_and_ir_batches_produce_same_target_for_same_seed() -> None:
    text_batch, text_targets = graph_batch(protocol=TextProtocol(), seeds=[91], entity_count=8, device="cpu")
    ir_batch, ir_targets = graph_batch(protocol=StructuredProtocol(), seeds=[91], entity_count=8, device="cpu")
    assert text_targets.tolist() == ir_targets.tolist()
    assert text_batch.bit_costs.item() != ir_batch.bit_costs.item()


def test_tiny_training_loop_updates_model_and_evaluates() -> None:
    config = GraphTrainConfig(protocol="structured", entity_count=4, batch_size=2, train_steps=2, reasoning_steps=1, learning_rate=1e-3, eval_episodes=4)
    model, metrics = train_graph_baseline(config, model_kwargs=_tiny_kwargs(), device="cpu")
    assert isinstance(model, GraphBaselineModel)
    assert metrics["final_loss"] >= 0.0
    assert 0.0 <= metrics["eval_accuracy"] <= 1.0
