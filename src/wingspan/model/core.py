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
from wingspan.model import hand_model, mlp

if typing.TYPE_CHECKING:
    from wingspan.training import runmeta


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

    # Constant buffers registered in __init__ (declared here so the type checker
    # sees them as tensors rather than nn.Module's generic attribute access).
    # ``card_summary_matrix`` is registered only under ``tray_set_embedding``.
    card_features: torch.Tensor
    card_pad_mask: torch.Tensor
    card_summary_matrix: torch.Tensor

    def __init__(
        self,
        *,
        state_dim: int | None = None,
        choice_dim: int | None = None,
        num_families: int | None = None,
        arch: architecture.ModelArchitecture | None = None,
        spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
    ):
        super().__init__()
        # Lazily-filled cache of the inference card table (see ``card_table`` /
        # ``_card_table_for_pass``). ``None`` whenever the encoder weights or the
        # train/eval mode may have changed; recomputed on the next eval forward.
        # Set before any buffer registration / ``_apply`` could touch it.
        self._inference_card_table: torch.Tensor | None = None
        # ``spec`` selects the config-driven encoding shape (whether setup is in
        # the main model). Dims default to that spec's sizes; callers that pass
        # explicit dims (e.g. ``from_model_config``) must pass a matching spec.
        if state_dim is None:
            state_dim = encode.state_size(spec)
        if choice_dim is None:
            choice_dim = encode.choice_feature_dim(spec)
        if num_families is None:
            num_families = len(decisions.active_decision_families(spec.include_setup))
        if arch is None:
            arch = architecture.ModelArchitecture()
        self.spec = spec
        self.include_setup = spec.include_setup
        self.state_dim = state_dim
        self.choice_dim = choice_dim
        self.num_families = num_families
        self.arch = arch
        self.card_embed_dim = arch.card_embed_dim
        self.trunk_hidden = arch.trunk_embed_width  # M — kept for external readouts

        # The shared card encoder. Each card's fixed feature row — its static
        # attributes concatenated with its identity one-hot — is mapped by this MLP
        # to a ``card_embed_dim`` vector; stacking all cards' rows yields the
        # ``[181, card_embed_dim]`` card table (``card_table``) weight-shared across
        # every board slot, tray slot, the hand (mean-pooled), and each choice
        # candidate. Because the input is constant per card, the table is a pure
        # function of identity — a real (optionally nonlinear) model in training,
        # collapsible to a plain lookup at inference. A single card therefore has
        # one representation, derived from both its attributes and a learned
        # per-card component, used by the critic-state read and the actor's
        # candidate scoring alike (TRAINING.md §6.3: the table is the per-card
        # power readout the project's card-ranking goal needs). The card
        # feature matrix and the padding-row mask are constant buffers rebuilt from
        # the catalog (``persistent=False`` keeps them out of the checkpoint).
        self.card_encoder, _ = mlp.build_body(
            encode.CARD_FEATURE_DIM,
            arch.card_encoder_layers + (arch.card_embed_dim,),
            activation=arch.activation,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
            final_activation=False,
        )
        self.register_buffer(
            "card_features",
            torch.tensor(encode.card_feature_matrix(), dtype=torch.float32),
            persistent=False,
        )
        pad_mask = torch.ones(encode.HAND_MULTIHOT_DIM + 1, 1)
        pad_mask[0] = 0.0
        self.register_buffer("card_pad_mask", pad_mask, persistent=False)

        # Optional distinct hand encoder. When enabled it takes a card *set*'s
        # representation — [multi-hot (180) ⊕ set-summary (10)] = 190 dims — and
        # outputs a ``hand_embed_width``-wide set embedding. For the hand itself
        # the 10-dim hand-summary is redirected from the trunk's continuous input
        # into this encoder, so the trunk sees a correspondingly narrower feed.
        if arch.use_distinct_hand_model:
            self.hand_encoder, _ = mlp.build_body(
                encode.HAND_ENCODER_INPUT_DIM,
                arch.hand_encoder_layers + (arch.hand_embed_width,),
                activation=arch.activation,
                dropout=arch.dropout,
                layernorm=arch.layernorm,
                final_activation=False,
            )
        # Under ``tray_set_embedding`` the hand encoder also embeds the face-up
        # tray as a set; the per-card summary table lets ``_embed_state`` derive
        # the tray's multi-hot + summary from its three index columns
        # (``persistent=False`` keeps the constant out of the checkpoint).
        if arch.tray_set_embedding:
            self.register_buffer(
                "card_summary_matrix",
                torch.tensor(encode.card_summary_matrix(), dtype=torch.float32),
                persistent=False,
            )

        # The trunk reads the continuous state features plus the looked-up card
        # embeddings: the index block becomes one embedding per slot (flattened),
        # the hand multi-hot becomes either a mean-pooled card embedding (default)
        # or a dedicated hand encoder output (when use_distinct_hand_model is on),
        # and ``tray_set_embedding`` appends one tray-set embedding.
        # It keeps an activation on its final layer (its output is an internal
        # representation, not a logit); the choice encoder does not.
        trunk_in_dim = encode.trunk_input_dim(
            state_dim,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_embed_dim=arch.hand_embed_dim,
            tray_set_embedding=arch.tray_set_embedding,
        )
        self.state_trunk, _ = mlp.build_body(
            trunk_in_dim,
            arch.trunk_layers,
            activation=arch.activation,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
            final_activation=True,
        )
        # The per-choice encoder reads the candidate's non-identity features plus
        # its card identity embedded through the same shared table.
        choice_in_dim = encode.choice_input_dim(choice_dim, arch.card_embed_dim)
        self.choice_encoder, _ = mlp.build_body(
            choice_in_dim,
            arch.choice_layers,
            activation=arch.activation,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
            final_activation=False,
        )
        # One scoring head per judgment family, each a readout MLP over the M+N
        # concat. The trunk + choice-encoder are shared; specialization lives
        # here. ``family_idx`` routes each decision to its head in ``forward``.
        scorer_in_dim = arch.trunk_embed_width + arch.choice_embed_width
        self.scorers = nn.ModuleList(
            mlp.build_readout(
                scorer_in_dim,
                arch.head_layers_for(family_index),
                activation=arch.activation,
                dropout=arch.dropout,
            )
            for family_index in range(num_families)
        )
        # The value head reads the trunk context (a property of the board, not of
        # the decision asked, so it is shared across families).
        self.value_head = mlp.build_readout(
            arch.trunk_embed_width,
            arch.value_layers,
            activation=arch.activation,
            dropout=arch.dropout,
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
            spec=encode.EncodingSpec(include_setup=descriptor.include_setup),
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
        # The shared card table, threaded into both the state and choice embeds.
        # In training this is recomputed every pass (the encoder weights are
        # learning and must stay in the autograd graph); at inference the weights
        # are frozen between loads, so it is computed once and reused as a plain
        # lookup (``_card_table_for_pass``) instead of once per decision.
        card_table = self._card_table_for_pass()  # (181, card_embed_dim)

        # State trunk produces both the per-decision context and the value. The
        # flat state's card-identity columns are embedded through the shared
        # table before the trunk sees them.
        state_ctx = self.state_trunk(self._embed_state(state, card_table))  # (B, H)
        value = self.value_head(state_ctx).squeeze(-1)  # (B,)

        # Per-choice MLP. choices is (B, K, F); the Linear layers broadcast across
        # the K dimension naturally. Each candidate's card identity is embedded
        # through the same shared table first.
        ce = self.choice_encoder(self._embed_choices(choices, card_table))  # (B, K, H)
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

    def card_table(self) -> torch.Tensor:
        """The shared ``[181, card_embed_dim]`` card table: the constant card-feature
        matrix mapped through the card encoder, with the padding row (index 0)
        forced to zero so an empty board slot / padding candidate contributes a zero
        vector (restoring the old ``padding_idx=0`` contract the encoder's bias would
        otherwise break).

        Public because it is the model's per-card representation readout: at
        inference the weights are fixed, so this can be computed once and reused as a
        plain lookup, and it doubles as the ``[bird_index + 1] -> vector`` table the
        card-power analysis reads (TRAINING.md §6.3). ``forward`` reaches it through
        ``_card_table_for_pass`` (which caches it during inference)."""
        return self.card_encoder(self.card_features) * self.card_pad_mask

    def train(self, mode: bool = True) -> "PolicyValueNet":
        """Flip train/eval mode, invalidating the cached inference card table.

        This is the cache's invalidation point. Every way the encoder weights can
        change is bracketed by a mode flip through here: an optimizer step happens
        in training (the cache is bypassed there) and is followed by ``eval()``
        before inference; a weight reload (collection workers ``load_state_dict``,
        then ``eval()``; the eval harness likewise) is always followed by
        ``eval()``. ``eval()`` is ``train(False)``, so it routes through here and
        drops any table cached under the old weights."""
        self._inference_card_table = None
        return super().train(mode)

    def _card_table_for_pass(self) -> torch.Tensor:
        """The card table for one forward pass: recomputed every call in training
        (the encoder weights are learning, so the table must stay in the autograd
        graph), but computed once and memoized at inference, where the weights are
        frozen between loads. The cache is dropped on every train/eval flip
        (``train``) and rebuilt from the on-device card buffers, so it always
        lands on the live device. Callers MUST ``eval()`` after loading new
        weights (every weight-load path in the codebase does) so a stale table is
        never served."""
        if self.training:
            return self.card_table()
        cached = self._inference_card_table
        if cached is None:
            cached = self.card_table().detach()
            self._inference_card_table = cached
        return cached

    def _embed_state(
        self, state: torch.Tensor, card_table: torch.Tensor
    ) -> torch.Tensor:
        """Turn the flat state ``(B, state_dim)`` into the trunk's input by
        replacing the card-identity columns with shared card vectors: the index
        block becomes one ``card_table`` row per slot (flattened), and the hand
        multi-hot becomes a hand embedding (mean-pooled or from a dedicated encoder),
        both concatenated with the continuous features.

        When ``use_distinct_hand_model`` is on the 10-dim hand-summary stripe is
        removed from the continuous block and redirected into the hand encoder,
        so the trunk's continuous feed is correspondingly narrower. When
        ``tray_set_embedding`` is also on, one tray-*set* embedding is appended:
        the three tray index columns are turned into a derived multi-hot + set
        summary and passed through the same hand encoder (the tray's three
        per-slot ``card_table`` rows in ``slot_emb`` are untouched)."""
        off_index = encode.OFF_CARD_INDEX
        off_hand = encode.OFF_HAND_MULTIHOT
        off_decision = encode.OFF_DECISION_TYPE

        # Card-index block -> per-slot card-table lookups, flattened. The encoder
        # always writes indices in range; the clamp only guards synthetic inputs.
        card_idx = (
            state[:, off_index:off_hand].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
        )
        slot_emb = card_table[card_idx].reshape(card_idx.shape[0], -1)

        hand_multihot = state[:, off_hand:off_decision]

        if self.arch.use_distinct_hand_model:
            # Strip the 10-dim hand summary from the continuous prefix and feed
            # it together with the multi-hot into the dedicated hand encoder.
            hand_sum_off = encode.HAND_SUMMARY_OFFSET
            hand_sum_end = hand_sum_off + encode.HAND_SUMMARY_DIM
            prefix = state[:, :off_index]
            continuous = torch.cat(
                [
                    prefix[:, :hand_sum_off],
                    prefix[:, hand_sum_end:],
                    state[:, off_decision:],
                ],
                dim=-1,
            )
            hand_summary = state[:, hand_sum_off:hand_sum_end]
            hand_emb = hand_model.embed_card_set(
                self.hand_encoder, hand_multihot, hand_summary
            )
        else:
            continuous = torch.cat(
                [state[:, :off_index], state[:, off_decision:]], dim=-1
            )
            # Hand multi-hot -> mean of held cards' vectors (rows 1.. skip padding).
            hand_sum = hand_multihot @ card_table[1:]
            hand_count = hand_multihot.sum(dim=-1, keepdim=True).clamp(min=1.0)
            hand_emb = hand_sum / hand_count

        if not self.arch.tray_set_embedding:
            return torch.cat([continuous, slot_emb, hand_emb], dim=-1)

        # Tray-set embedding: the trailing TRAY_SIZE index columns become a
        # derived multi-hot + set summary, embedded through the shared hand
        # encoder as one more set vector beside the per-slot lookups.
        tray_idx = card_idx[:, encode.N_BOARD_INDEX_SLOTS :]
        tray_multihot = hand_model.multihot_from_indices(
            tray_idx, encode.HAND_MULTIHOT_DIM
        )
        tray_summary = hand_model.set_summary_from_indices(
            tray_idx, self.card_summary_matrix
        )
        tray_set_emb = hand_model.embed_card_set(
            self.hand_encoder, tray_multihot, tray_summary
        )
        return torch.cat([continuous, slot_emb, tray_set_emb, hand_emb], dim=-1)

    def _embed_choices(
        self, choices: torch.Tensor, card_table: torch.Tensor
    ) -> torch.Tensor:
        """Turn the per-choice features ``(B, K, choice_dim)`` into the choice
        encoder's input by mapping the candidate's two card regions through the
        shared card table and concatenating them with the remaining features.

        The 15-slot board-index block (the board_target stripe's occupant ids)
        sits immediately before the candidate bird one-hot; the board indices
        become one ``card_embed_dim`` vector per slot (flattened) and the
        candidate one-hot becomes a single ``card_embed_dim`` vector (a setup
        pick's kept-set multi-hot sums their vectors). Everything else — including
        bonus_id and any trailing setup_agg stripe — passes through. These offsets
        are spec-invariant, so the slice is config-independent."""
        off_board = encode.CHOICE_BOARD_IDX_OFFSET
        off_bird = encode.CHOICE_BIRD_ID_OFFSET  # == off_board + board-index slots
        end_bird = off_bird + encode.CHOICE_BIRD_ID_DIM
        board_idx = (
            choices[..., off_board:off_bird].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
        )
        bird_multihot = choices[..., off_bird:end_bird]
        rest = torch.cat([choices[..., :off_board], choices[..., end_bird:]], dim=-1)
        cand_emb = bird_multihot @ card_table[1:]
        board_emb = card_table[board_idx].reshape(*board_idx.shape[:-1], -1)
        return torch.cat([rest, cand_emb, board_emb], dim=-1)
