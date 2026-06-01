"""PyTorch model: pointer-style actor-critic with per-family policy heads.

The network scores each candidate at a decision point individually rather
than emitting a fixed-slot policy head. At each forward pass the caller
provides:

* ``state``      — ``(B, state_dim)`` POV-aware game features
* ``choices``    — ``(B, K, choice_dim)`` per-choice features (padded)
* ``mask``       — ``(B, K)`` 1.0 for real choices, 0.0 for padding
* ``family_idx`` — ``(B,)`` long; the judgment-family head index for each
  decision (``decisions.family_index_for``)

The trunk reads the state, a separate MLP reads each choice, the two are
concatenated and passed through the scoring head **for that decision's
judgment family** to produce one logit per candidate. The trunk and the
per-choice encoder are shared across all families; only the final scorer
specializes (one head per ``decisions.DecisionFamily``), and the value head
is shared because position value is a property of the board, not of the
decision being asked. Padding rows get ``-inf`` so they never receive
probability mass.

This realizes DECISIONS.md §0's "shared trunk + per-family heads": the single
monolithic scorer is replaced by a family-routed bank of scorers, turning
"one model conditioned on a decision-type one-hot" into "one model per kind
of choice" without multiplying the trunk or starving the shared critic.
"""

from __future__ import annotations

import typing

import torch
from torch import nn

from wingspan import architecture, decisions, encode

if typing.TYPE_CHECKING:
    from wingspan.training import runmeta

# The selectable activation functions, keyed by their descriptor enum. Each maps
# to a zero-argument ``nn.Module`` factory.
_ACTIVATIONS: dict[architecture.ActivationName, typing.Callable[[], nn.Module]] = {
    architecture.ActivationName.RELU: nn.ReLU,
    architecture.ActivationName.GELU: nn.GELU,
    architecture.ActivationName.TANH: nn.Tanh,
    architecture.ActivationName.SILU: nn.SiLU,
    architecture.ActivationName.LEAKY_RELU: nn.LeakyReLU,
}


class PolicyValueNet(nn.Module):
    """Actor-critic over (state, choice-set) decisions with per-family heads.

    A state trunk feeds (a) a state-context vector used to rescore every
    candidate and (b) the shared value head; a per-choice encoder consumes the
    per-choice features. The score for choice ``i`` is an MLP over
    ``concat(state_ctx, choice_emb[i])``, selected per decision by its judgment
    family (``decisions.ALL_DECISION_FAMILIES``) so different kinds of choice are
    scored by different, specialized heads.

    Every block's depth, width, activation, dropout, and LayerNorm come from a
    :class:`architecture.ModelArchitecture` (the topology saved to
    ``model_config.json``), so the network's shape is fully data-driven. The
    trunk ends at width ``M`` and the choice encoder at width ``N``; their
    outputs are concatenated to ``M+N`` for the scorer heads.
    """

    def __init__(
        self,
        *,
        state_dim: int | None = None,
        choice_dim: int = encode.CHOICE_FEATURE_DIM,
        num_families: int = len(decisions.ALL_DECISION_FAMILIES),
        arch: architecture.ModelArchitecture | None = None,
    ):
        super().__init__()
        if state_dim is None:
            state_dim = encode.state_size()
        if arch is None:
            arch = architecture.ModelArchitecture()
        self.state_dim = state_dim
        self.choice_dim = choice_dim
        self.num_families = num_families
        self.arch = arch
        self.card_embed_dim = arch.card_embed_dim
        self.trunk_hidden = arch.trunk_embed_width  # M — kept for external readouts

        # One shared learned vector per core-set bird (row 0 is the padding /
        # empty-slot vector). nn.Embedding(idx) == one_hot @ W, so this is the
        # per-position card encoder, weight-shared across every board slot, tray
        # slot, the hand (mean-pooled), and each choice candidate. A single card
        # therefore has one learned value used by both the critic-state read and
        # the actor's candidate scoring (DECISIONS.md card power-ranking goal).
        self.card_embed = nn.Embedding(
            encode.HAND_MULTIHOT_DIM + 1, arch.card_embed_dim, padding_idx=0
        )

        # The trunk reads the continuous state features plus the looked-up card
        # embeddings: the index block becomes one embedding per slot (flattened),
        # the hand multi-hot becomes a single mean-pooled embedding. It keeps an
        # activation on its final layer (its output is an internal representation,
        # not a logit); the choice encoder does not (its output is concatenated
        # with the trunk context before scoring, matching the original shape).
        trunk_in_dim = (
            state_dim
            - encode.N_CARD_INDEX_SLOTS  # index columns -> per-slot embeddings
            - encode.HAND_MULTIHOT_DIM  # hand multi-hot -> one pooled embedding
            + encode.N_CARD_INDEX_SLOTS * arch.card_embed_dim
            + arch.card_embed_dim
        )
        self.state_trunk, _ = _build_body(
            trunk_in_dim, arch.trunk_layers, arch, final_activation=True
        )
        # The per-choice encoder reads the candidate's non-identity features plus
        # its card identity embedded through the same shared table.
        choice_in_dim = choice_dim - encode.CHOICE_BIRD_ID_DIM + arch.card_embed_dim
        self.choice_encoder, _ = _build_body(
            choice_in_dim, arch.choice_layers, arch, final_activation=False
        )
        # One scoring head per judgment family, each a readout MLP over the M+N
        # concat. The trunk + choice-encoder are shared; specialization lives
        # here. ``family_idx`` routes each decision to its head in ``forward``.
        scorer_in_dim = arch.trunk_embed_width + arch.choice_embed_width
        self.scorers = nn.ModuleList(
            _build_readout(scorer_in_dim, arch.head_layers, arch)
            for _ in range(num_families)
        )
        # The value head reads the trunk context (a property of the board, not of
        # the decision asked, so it is shared across families).
        self.value_head = _build_readout(
            arch.trunk_embed_width, arch.value_layers, arch
        )

    @classmethod
    def from_model_config(cls, descriptor: "runmeta.ModelConfig") -> "PolicyValueNet":
        """Rebuild a net matching a saved ``model_config.json`` descriptor — its
        full topology plus the encoding dims and family-head count it was trained
        under. The returned net has fresh weights in the saved shape, ready for
        ``load_state_dict`` from the run's checkpoint."""
        return cls(
            state_dim=descriptor.state_dim,
            choice_dim=descriptor.choice_dim,
            num_families=len(descriptor.family_order),
            arch=descriptor.architecture,
        )

    def forward(
        self,
        state: torch.Tensor,
        choices: torch.Tensor,
        mask: torch.Tensor,
        family_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score every candidate at every decision in the batch.

        Args:
            state:      ``(B, state_dim)``
            choices:    ``(B, K, choice_dim)``  — pad rows are arbitrary
            mask:       ``(B, K)`` with 1.0 on real choices, 0.0 on padding.
            family_idx: ``(B,)`` long — judgment-family head index per decision
                (``decisions.family_index_for``); each value in
                ``[0, num_families)``.

        Returns:
            logits: ``(B, K)`` — masked rows are set to ``-inf``
            value:  ``(B,)``
        """
        # State trunk produces both the per-decision context and the value. The
        # flat state's card-identity columns are embedded through the shared
        # table before the trunk sees them.
        state_ctx = self.state_trunk(self._embed_state(state))  # (B, H)
        value = self.value_head(state_ctx).squeeze(-1)  # (B,)

        # Per-choice MLP. choices is (B, K, F); the Linear layers broadcast across
        # the K dimension naturally. Each candidate's card identity is embedded
        # through the same shared table first.
        ce = self.choice_encoder(self._embed_choices(choices))  # (B, K, H)
        num_choices = ce.shape[1]
        s_exp = state_ctx.unsqueeze(1).expand(-1, num_choices, -1)  # (B, K, H)
        combined = torch.cat([s_exp, ce], dim=-1)  # (B, K, M+N)

        # Route each decision through its judgment family's scoring head. Every
        # candidate in a decision shares one head (family is a property of the
        # decision, not the candidate), so we slice the batch by family, score
        # each slice with its head, and scatter the logits back. Disjoint row
        # sets cover the whole batch, so every row is scored exactly once.
        scores = combined.new_zeros(combined.shape[:2])  # (B, K)
        for family in range(self.num_families):
            rows = family_idx == family
            if not bool(rows.any()):
                continue
            scores = scores.index_copy(
                0,
                rows.nonzero(as_tuple=True)[0],
                self.scorers[family](combined[rows]).squeeze(-1),
            )

        # Mask out padding. Use very-negative rather than -inf to avoid NaN
        # if a row turns out to be entirely padded (defensive — shouldn't
        # happen in practice). For all-real rows -inf is fine; we use a
        # large finite number so softmax stays numerically clean either way.
        neg_inf = torch.full_like(scores, float("-inf"))
        logits = torch.where(mask > 0.5, scores, neg_inf)
        # If a whole row is masked (no real choices), fall back to a zero
        # row so downstream log_softmax doesn't produce NaN. Caller should
        # never feed an empty decision.
        any_legal = mask.sum(dim=-1, keepdim=True) > 0
        logits = torch.where(any_legal, logits, torch.zeros_like(logits))
        return logits, value

    def _embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """Turn the flat state ``(B, state_dim)`` into the trunk's input by
        replacing the card-identity columns with shared embeddings: the index
        block becomes one embedding per slot (flattened) and the hand multi-hot
        becomes a single mean-pooled embedding, both concatenated with the
        continuous features (everything outside the index/hand blocks, including
        the trailing decision-type stripe)."""
        off_index = encode.OFF_CARD_INDEX
        off_hand = encode.OFF_HAND_MULTIHOT
        off_decision = encode.OFF_DECISION_TYPE
        continuous = torch.cat([state[:, :off_index], state[:, off_decision:]], dim=-1)

        # Card-index block -> per-slot embedding lookups, flattened. The encoder
        # always writes indices in range; the clamp only guards synthetic inputs.
        card_idx = (
            state[:, off_index:off_hand].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
        )
        slot_emb = self.card_embed(card_idx).reshape(card_idx.shape[0], -1)

        # Hand multi-hot -> mean of held cards' embeddings via the same weight.
        hand_multihot = state[:, off_hand:off_decision]
        hand_sum = hand_multihot @ self.card_embed.weight[1:]
        hand_count = hand_multihot.sum(dim=-1, keepdim=True).clamp(min=1.0)
        hand_emb = hand_sum / hand_count

        return torch.cat([continuous, slot_emb, hand_emb], dim=-1)

    def _embed_choices(self, choices: torch.Tensor) -> torch.Tensor:
        """Turn the per-choice features ``(B, K, choice_dim)`` into the choice
        encoder's input by embedding each candidate's card-identity stripe through
        the shared table (a single-card one-hot maps to that card's embedding; the
        setup pick's kept-set multi-hot sums their embeddings) and concatenating
        the candidate's remaining, non-identity features."""
        off_bird = encode.CHOICE_BIRD_ID_OFFSET
        end_bird = off_bird + encode.CHOICE_BIRD_ID_DIM
        bird_multihot = choices[..., off_bird:end_bird]
        rest = torch.cat([choices[..., :off_bird], choices[..., end_bird:]], dim=-1)
        cand_emb = bird_multihot @ self.card_embed.weight[1:]
        return torch.cat([rest, cand_emb], dim=-1)


###### PRIVATE #######


def _activation_module(name: architecture.ActivationName) -> nn.Module:
    """A fresh activation module for the descriptor's chosen function."""
    return _ACTIVATIONS[name]()


def _build_body(
    in_dim: int,
    widths: architecture.Widths,
    arch: architecture.ModelArchitecture,
    *,
    final_activation: bool,
) -> tuple[nn.Sequential, int]:
    """Build a body MLP — the trunk or the choice encoder — and return it with
    its output width. Each layer is ``Linear`` → (optional) ``LayerNorm`` →
    activation → (optional) ``Dropout``; the activation + dropout on the final
    layer are emitted only when ``final_activation`` (the trunk keeps a trailing
    activation, the choice encoder does not). LayerNorm — when enabled on the
    architecture — is applied to these body blocks; the readout heads omit it."""
    modules: list[nn.Module] = []
    prev = in_dim
    last_index = len(widths) - 1
    for index, width in enumerate(widths):
        modules.append(nn.Linear(prev, width))
        if arch.layernorm:
            modules.append(nn.LayerNorm(width))
        if final_activation or index != last_index:
            modules.append(_activation_module(arch.activation))
            if arch.dropout > 0.0:
                modules.append(nn.Dropout(arch.dropout))
        prev = width
    return nn.Sequential(*modules), prev


def _build_readout(
    in_dim: int,
    widths: architecture.Widths,
    arch: architecture.ModelArchitecture,
) -> nn.Sequential:
    """Build a scalar-readout MLP (a scorer head or the value head): the hidden
    ``widths`` as ``Linear`` → activation → (optional) ``Dropout`` blocks, then a
    final ``Linear(·, 1)`` with no activation. Empty ``widths`` collapses to a
    single ``Linear(in_dim, 1)`` — the original head shapes (scorer ``2H→H→1``,
    value head ``H→1``) when the defaults are used."""
    modules: list[nn.Module] = []
    prev = in_dim
    for width in widths:
        modules.append(nn.Linear(prev, width))
        modules.append(_activation_module(arch.activation))
        if arch.dropout > 0.0:
            modules.append(nn.Dropout(arch.dropout))
        prev = width
    modules.append(nn.Linear(prev, 1))
    return nn.Sequential(*modules)
