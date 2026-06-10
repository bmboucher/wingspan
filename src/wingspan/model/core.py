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
    import numpy as np

    from wingspan import state
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

        # Build all submodules in registration order — this order is the
        # state_dict key order and must be kept byte-identical across runs
        # (checkpoint compat). Each _build_* method registers exactly the
        # submodules named in the method's docstring.
        self._build_card_encoder(arch)
        self._build_hand_encoder(arch)
        self._build_trunk(state_dim, arch)
        self._build_choice_encoder(choice_dim, arch)
        self._build_scorers(arch, num_families)
        self._build_value_head(arch)

    @classmethod
    def class_for_version(cls, artifact_version: str) -> "type[PolicyValueNet]":
        """The net class whose frozen geometry matches ``artifact_version``.

        The single era-routing table: pre-0.1 → ``v0_0.PolicyValueNetV00``
        (frozen choice encoding); 0.1 → ``v0_1.PolicyValueNetV01`` (frozen
        229-wide card encoder); 0.2 → ``v0_2.PolicyValueNetV02`` (frozen
        771-dim state geometry); current era → the live class. Used by every
        construction seam that must honor an artifact's era — the checkpoint
        loaders, ``from_model_config``, and the era-pinned training pipeline
        (``TrainConfig.encoding_version``)."""
        from wingspan.compat import (  # local: compat subclasses this net
            v0_0,
            v0_1,
            v0_2,
        )

        if v0_0.uses_v0_0_choice_encoding(artifact_version):
            return v0_0.PolicyValueNetV00
        if v0_1.uses_v0_1_card_feature_encoding(artifact_version):
            return v0_1.PolicyValueNetV01
        if v0_2.uses_v0_2_state_encoding(artifact_version):
            return v0_2.PolicyValueNetV02
        return PolicyValueNet

    @classmethod
    def from_model_config(cls, descriptor: "runmeta.ModelConfig") -> "PolicyValueNet":
        """Rebuild a net matching a saved ``model_config.json`` descriptor — its
        full topology plus the encoding dims and family-head count it was trained
        under. The returned net has fresh weights in the saved shape, ready for
        ``load_state_dict`` from the run's checkpoint.

        The version routing (:meth:`class_for_version`) selects the right compat
        subclass so every caller gets a net whose geometry matches its weights
        without consulting the version."""
        net_cls = cls.class_for_version(descriptor.version)
        return net_cls(
            state_dim=descriptor.state_dim,
            choice_dim=descriptor.choice_dim,
            num_families=len(descriptor.family_order),
            arch=descriptor.architecture,
            spec=encode.EncodingSpec(include_setup=descriptor.include_setup),
        )

    def encode_state(
        self,
        game_state: "state.GameState",
        decision: decisions.Decision[typing.Any],
    ) -> "np.ndarray":
        """Featurize ``game_state`` at ``decision`` for one forward pass of
        *this* net. The net owns its encoding (``self.spec`` and, in compat
        subclasses, the artifact-era geometry), so inference call sites ask the
        net instead of pairing the live encoder with a spec by hand."""
        return encode.encode_state(game_state, decision, self.spec)

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: "state.GameState",
    ) -> "np.ndarray":
        """Featurize every choice in ``decision`` for one forward pass of this
        net (see :meth:`encode_state`). Compat subclasses override this with
        their frozen era's encoder."""
        return encode.encode_choices(decision, game_state, self.spec)

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

    def _build_card_encoder(self, arch: architecture.ModelArchitecture) -> None:
        """Register ``card_encoder``, ``card_features``, ``card_pad_mask``.

        Each card's fixed feature row — its static attributes concatenated with
        its identity one-hot — is mapped by this MLP to a ``card_embed_dim``
        vector. Because the input is constant per card, the table is a pure
        function of identity, collapsible to a plain lookup at inference
        (TRAINING.md §6.3). The constant buffers are ``persistent=False``."""
        self.card_encoder, _ = mlp.build_body(
            encode.CARD_FEATURE_DIM,
            arch.card_encoder_layers + (arch.card_embed_dim,),
            activation=arch.activation,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
            final_activation=arch.encoder_final_activation,
        )
        self.register_buffer(
            "card_features",
            torch.tensor(encode.card_feature_matrix(), dtype=torch.float32),
            persistent=False,
        )
        pad_mask = torch.ones(encode.HAND_MULTIHOT_DIM + 1, 1)
        pad_mask[0] = 0.0
        self.register_buffer("card_pad_mask", pad_mask, persistent=False)

    def _build_hand_encoder(self, arch: architecture.ModelArchitecture) -> None:
        """Register ``hand_encoder`` and ``card_summary_matrix`` (both conditional).

        When ``use_distinct_hand_model`` is on, a dedicated MLP encodes a card
        set's [multi-hot ⊕ set-summary] representation to ``hand_embed_width``
        dims. When ``tray_set_embedding`` is also on, the per-card summary
        table lets ``_embed_state`` derive tray multi-hots from index columns."""
        if arch.use_distinct_hand_model:
            self.hand_encoder, _ = mlp.build_body(
                encode.HAND_ENCODER_INPUT_DIM,
                arch.hand_encoder_layers + (arch.hand_embed_width,),
                activation=arch.activation,
                dropout=arch.dropout,
                layernorm=arch.layernorm,
                final_activation=arch.encoder_final_activation,
            )
        if arch.tray_set_embedding:
            self.register_buffer(
                "card_summary_matrix",
                torch.tensor(encode.card_summary_matrix(), dtype=torch.float32),
                persistent=False,
            )

    def _build_trunk(
        self, state_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Register ``state_trunk``.

        The trunk reads continuous state features plus looked-up card embeddings
        (index block → one embedding per slot, hand → mean-pool or dedicated
        encoder, tray set when enabled). Always keeps a final activation — its
        output is an internal representation consumed by both the value head and
        the scorer concat."""
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

    def _build_choice_encoder(
        self, choice_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Register ``choice_encoder``.

        The per-choice encoder reads each candidate's non-identity features plus
        its card identity embedded through the shared card table. Applies a final
        activation when ``arch.encoder_final_activation`` is True."""
        choice_in_dim = encode.choice_input_dim(
            choice_dim, arch.card_embed_dim, include_setup=self.include_setup
        )
        self.choice_encoder, _ = mlp.build_body(
            choice_in_dim,
            arch.choice_layers,
            activation=arch.activation,
            dropout=arch.dropout,
            layernorm=arch.layernorm,
            final_activation=arch.encoder_final_activation,
        )

    def _build_scorers(
        self, arch: architecture.ModelArchitecture, num_families: int
    ) -> None:
        """Register ``scorers`` (one head per judgment family).

        Each head is a readout MLP over the M+N trunk/choice concat.
        ``family_idx`` routes each decision to its head in ``forward``."""
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

    def _build_value_head(self, arch: architecture.ModelArchitecture) -> None:
        """Register ``value_head``.

        The value head reads the trunk context (a property of the board, not of
        the decision asked) and is shared across all judgment families."""
        self.value_head = mlp.build_readout(
            arch.trunk_embed_width,
            arch.value_layers,
            activation=arch.activation,
            dropout=arch.dropout,
        )

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

    def _state_embed_offsets(self) -> tuple[int, int, int]:
        """The ``(card-index, hand-multi-hot, decision-type)`` slice offsets that
        :meth:`_embed_state` splits the flat state vector on.

        The live net reads them from the current ``encode.layout`` chain. Era
        compat nets carry their own frozen-geometry state vector (e.g. the
        771-dim pre-0.3 vector), so they override this to return the offsets
        that vector was written with — never the live ones. Slicing an old
        vector at live offsets is silent corruption: the widths can coincide
        (no crash) while the columns read are wrong, so this seam, not just
        ``encode_state``, must move with the era (see ``docs/VERSIONING.md`` and
        ``compat/INDEX.md``)."""
        return (
            encode.OFF_CARD_INDEX,
            encode.OFF_HAND_MULTIHOT,
            encode.OFF_DECISION_TYPE,
        )

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
        off_index, off_hand, off_decision = self._state_embed_offsets()

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
        encoder's input by mapping the candidate's card regions through the
        shared card table and concatenating them with the remaining features.

        The 15-slot board-index block (occupant ids plus the placement rows'
        landing-slot marker) sits immediately before the candidate bird-index
        column; the board indices become one ``card_embed_dim`` vector per slot
        (flattened, index 0 = the table's padding row) and the candidate index
        becomes a single ``card_embed_dim`` vector, explicitly zeroed when the
        column is 0 so a non-bird row contributes nothing. Those two offsets
        are spec-invariant; the trailing kept_multihot stripe (a setup pick's
        kept-set multi-hot) exists only when ``include_setup`` and is summed
        through the card table into one more vector. Everything else —
        including bonus_id and the setup_agg stripe — passes through."""
        off_board = encode.CHOICE_BOARD_IDX_OFFSET
        off_bird = encode.CHOICE_BIRD_ID_OFFSET  # == off_board + board-index slots
        end_bird = off_bird + encode.CHOICE_BIRD_ID_DIM
        board_idx = (
            choices[..., off_board:off_bird].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
        )
        cand_idx = (
            choices[..., off_bird].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
        )  # (B, K)
        cand_mask = (cand_idx > 0).unsqueeze(-1).to(card_table.dtype)
        cand_emb = card_table[cand_idx] * cand_mask
        board_emb = card_table[board_idx].reshape(*board_idx.shape[:-1], -1)
        if self.include_setup:
            off_kept = encode.CHOICE_KEPT_MULTIHOT_OFFSET
            kept_multihot = choices[..., off_kept:]
            kept_emb = kept_multihot @ card_table[1:]
            rest = torch.cat(
                [choices[..., :off_board], choices[..., end_bird:off_kept]], dim=-1
            )
            return torch.cat([rest, cand_emb, board_emb, kept_emb], dim=-1)
        rest = torch.cat([choices[..., :off_board], choices[..., end_bird:]], dim=-1)
        return torch.cat([rest, cand_emb, board_emb], dim=-1)
