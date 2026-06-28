# pyright: reportPrivateUsage=false
# (white-box tests of the model's private inference card-table cache —
# they read ``_inference_card_table`` to assert the cache mechanism directly)
"""The inference card-table cache (``model.PolicyValueNet``).

``forward`` memoizes the shared card table during inference — the card encoder's
weights are frozen between loads, so recomputing it per decision is waste — and
recomputes it every pass during training, where it must stay in the autograd
graph. These tests lock in that the cache returns the *correct* table, never goes
stale across a weight load or a train/eval flip, and never severs the training
gradient path. See ``card-table-inference-cache`` in the perf notes for why this
is the dominant per-decision collection cost.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wingspan import encode, model


def _inputs(n_choices: int = 4, batch: int = 1):
    """A well-shaped (state, choices, mask, family) batch for one forward pass.

    Unannotated by design: ``torch`` arrives via ``pytest.importorskip`` (an
    ``Any``), so it cannot name types in an annotation — the existing model tests
    follow the same convention."""
    state = torch.randn(batch, encode.state_size())
    choices = torch.randn(batch, n_choices, encode.CHOICE_FEATURE_DIM)
    mask = torch.ones(batch, n_choices)
    family = torch.zeros(batch, dtype=torch.long)
    return state, choices, mask, family


def test_eval_cache_holds_a_fresh_card_table():
    """The first eval forward populates the cache with a value equal to a fresh
    ``card_table()`` recompute."""
    net = model.PolicyValueNet()
    net.eval()
    assert net._inference_card_table is None  # nothing cached before any forward
    fresh = net.card_table().detach()
    with torch.no_grad():
        net(*_inputs())
    cached = net._inference_card_table
    assert cached is not None
    assert torch.equal(cached, fresh)


def test_eval_matches_training_recompute():
    """Eval (cached table) and train (fresh recompute) forwards agree for the
    default architecture — proof the cached table is the *right* table. The
    default arch has no dropout / LayerNorm, so train and eval differ only in the
    card-table path."""
    net = model.PolicyValueNet()
    args = _inputs()
    net.train()
    with torch.no_grad():
        logits_train, value_train = net(*args)
    net.eval()
    with torch.no_grad():
        logits_eval, value_eval = net(*args)
    assert torch.allclose(logits_train, logits_eval, atol=1e-6)
    assert torch.allclose(value_train, value_eval, atol=1e-6)


def test_repeated_eval_forwards_are_identical():
    """Two eval forwards on the same inputs return identical outputs — the cache
    changes nothing observable, only how often the encoder runs."""
    net = model.PolicyValueNet()
    net.eval()
    args = _inputs()
    with torch.no_grad():
        first = net(*args)
        second = net(*args)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])


def test_weight_reload_then_eval_refreshes_cache():
    """A weight reload followed by ``eval()`` (the contract every loader honors)
    refreshes the table, so the next forward reflects the freshly-loaded weights
    instead of serving a table cached under the old ones."""
    net = model.PolicyValueNet()
    net.eval()
    other = model.PolicyValueNet()  # independent random weights
    args = _inputs()
    with torch.no_grad():
        before, _ = net(*args)  # caches the original table
    net.load_state_dict(other.state_dict())
    net.eval()  # the mandatory post-load eval() drops the stale table
    with torch.no_grad():
        after, _ = net(*args)
    assert not torch.allclose(before, after), "stale cache served after weight load"

    # And the reloaded net agrees with a net built fresh from those weights.
    reference = model.PolicyValueNet()
    reference.load_state_dict(other.state_dict())
    reference.eval()
    with torch.no_grad():
        reference_logits, _ = reference(*args)
    assert torch.allclose(after, reference_logits, atol=1e-6)


def test_mode_flip_invalidates_cache():
    """A train/eval flip clears the cache (it either follows a weight reload or
    re-enables training); eval also clears it, since ``eval()`` is ``train(False)``."""
    net = model.PolicyValueNet()
    net.eval()
    with torch.no_grad():
        net(*_inputs())
    assert net._inference_card_table is not None
    net.train()
    assert net._inference_card_table is None
    net.eval()
    assert net._inference_card_table is None


def test_training_forward_keeps_card_encoder_in_graph():
    """In training the card table is recomputed every pass and the loss
    backpropagates into the card encoder — caching must not sever that path, and
    training must never populate the inference cache."""
    net = model.PolicyValueNet()
    net.train()
    logits, value = net(*_inputs())
    (logits.sum() + value.sum()).backward()
    card_grads = [param.grad for param in net.card_encoder.parameters()]
    assert any(
        grad is not None and bool(torch.any(grad != 0.0)) for grad in card_grads
    ), "card encoder received no gradient — the training graph was severed"
    assert net._inference_card_table is None
