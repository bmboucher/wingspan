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


class StateEmbedOffsets(typing.NamedTuple):
    """The era-dependent column offsets :meth:`PolicyValueNet._embed_state` slices
    the flat state vector on.

    The live net derives them from the current ``encode.layout`` chain; a
    frozen-geometry compat net overrides
    :meth:`PolicyValueNet._state_embed_offsets` to return the offsets *its* era's
    vector was written with. The four fields do not all shift by one uniform delta
    between eras: ``card_index`` / ``hand_multihot`` / ``decision_type`` follow the
    misc-scalars stripe, while ``hand_summary`` precedes it, so a stripe inserted
    between them moves only a subset. Every offset ``_embed_state`` reads lives
    here — not as a bare ``encode.*`` constant — so a shim freezes all of them at
    once and a new stripe cannot silently desync one (see ``docs/VERSIONING.md``
    and ``compat/INDEX.md``).

    ``hand_summary`` / ``hand_summary_end`` together describe the hand-summary slice
    in the state vector. In live v0.9+ encoding the stripe is absent and both are 0
    (the model derives the summary in-model from the multi-hot). In pre-0.9 shims
    both are set to the frozen position so ``_embed_state`` reads the stripe from the
    frozen vector and passes it to the hand encoder (the historical path)."""

    card_index: int
    hand_multihot: int
    decision_type: int
    hand_summary: int
    hand_summary_end: int


class ChoiceEmbedOffsets(typing.NamedTuple):
    """The era-dependent column offsets :meth:`PolicyValueNet._embed_choices` slices
    the flat choice row on.

    ``becomes_playable`` is ``None`` for pre-0.6 eras (the stripe did not exist);
    ``kept_multihot`` is ``None`` when ``include_setup`` is ``False``."""

    board_idx: int
    bird_id: int
    becomes_playable: int | None
    kept_multihot: int | None


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
    # Optional attention modules, registered only when use_board_attention is on.
    board_attn_me: nn.MultiheadAttention
    board_attn_opp: nn.MultiheadAttention

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
        self._build_board_attention(arch)
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
        771-dim state geometry); 0.3 → ``v0_3.PolicyValueNetV03`` (frozen
        790-dim state geometry, one-hot round + cubes); 0.4–0.5 →
        ``v0_4.PolicyValueNetV04`` (frozen 795-dim state + no becomes_playable
        choice stripe); 0.6 → ``v0_6.PolicyValueNetV06`` (frozen 224-wide card
        encoder, pre-or_cost, eggs-included food encoding); 0.7 →
        ``v0_7.PolicyValueNetV07`` (eggs-included food ``becomes_playable``);
        current era → the live class. Used by every construction seam that must
        honor an artifact's era."""
        from wingspan.compat import (  # local: compat subclasses this net
            v0_0,
            v0_1,
            v0_2,
            v0_3,
            v0_4,
            v0_6,
            v0_7,
            v0_8,
        )

        if v0_0.uses_v0_0_choice_encoding(artifact_version):
            return v0_0.PolicyValueNetV00
        if v0_1.uses_v0_1_card_feature_encoding(artifact_version):
            return v0_1.PolicyValueNetV01
        if v0_2.uses_v0_2_state_encoding(artifact_version):
            return v0_2.PolicyValueNetV02
        if v0_3.uses_v0_3_state_encoding(artifact_version):
            return v0_3.PolicyValueNetV03
        if v0_4.uses_v0_4_encoding(artifact_version):
            return v0_4.PolicyValueNetV04
        if v0_6.uses_v0_6_card_feature_encoding(artifact_version):
            return v0_6.PolicyValueNetV06
        if v0_7.uses_v0_7_becomes_playable_encoding(artifact_version):
            return v0_7.PolicyValueNetV07
        if v0_8.uses_pre_v09_state_encoding(artifact_version):
            return v0_8.PolicyValueNetV08
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
            between_activation=arch.card_between_activation_resolved,
            final_activation=arch.card_final_activation_resolved,
            dropout=arch.card_dropout_resolved,
            layernorm=arch.card_layernorm_resolved,
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
        dims. The per-card summary table (``card_summary_matrix``) is registered
        whenever the distinct hand model is active — needed for the playability
        multi-hot blocks and, when ``tray_set_embedding`` is also on, the tray set."""
        if arch.use_distinct_hand_model:
            self.hand_encoder, _ = mlp.build_body(
                encode.HAND_ENCODER_INPUT_DIM,
                arch.hand_encoder_layers + (arch.hand_embed_width,),
                between_activation=arch.hand_between_activation_resolved,
                final_activation=arch.hand_final_activation_resolved,
                dropout=arch.hand_dropout_resolved,
                layernorm=arch.hand_layernorm_resolved,
            )
            self.register_buffer(
                "card_summary_matrix",
                torch.tensor(encode.card_summary_matrix(), dtype=torch.float32),
                persistent=False,
            )

    def _build_board_attention(self, arch: architecture.ModelArchitecture) -> None:
        """Register ``board_attn_me`` and ``board_attn_opp`` (conditional).

        When ``use_board_attention`` is on, two independent
        ``nn.MultiheadAttention`` modules are registered — one for the active
        player's board, one for the opponent's — each operating over 15 token
        slots of width ``card_embed_dim + SLOT_SCALAR_DIM``. Single-head only for
        this first pass (73 = prime for the default 64+9 token width; multi-head
        would require a projection)."""
        if not arch.use_board_attention:
            return
        token_dim = arch.card_embed_dim + encode.SLOT_SCALAR_DIM
        self.board_attn_me = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=1, batch_first=True
        )
        self.board_attn_opp = nn.MultiheadAttention(
            embed_dim=token_dim, num_heads=1, batch_first=True
        )

    def _build_trunk(
        self, state_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Register ``state_trunk``.

        The trunk reads continuous state features plus looked-up card embeddings
        (index block → one embedding per slot, hand → mean-pool or dedicated
        encoder, tray set when enabled, plus any extra playability multi-hot
        blocks). Always keeps a final activation — its output is an internal
        representation consumed by both the value head and the scorer concat."""
        offsets = self._state_embed_offsets()
        # Count extra hand-playability multi-hot blocks between hand_multihot and
        # decision_type. Live encoding has N_HAND_PLAYABLE_MULTIHOTS; pre-0.6
        # compat shims return the old decision_type offset so this is 0.
        n_extra = (
            offsets.decision_type - offsets.hand_multihot
        ) // encode.HAND_MULTIHOT_DIM - 1
        # In pre-0.9 frozen state vectors the hand-summary stripe is physically
        # present (hand_summary_end > hand_summary) and must be subtracted from the
        # continuous feed when building the trunk. In live v0.9+ it is derived
        # in-model so both fields are 0 and no subtraction is needed.
        hand_summary_in_state = offsets.hand_summary_end > offsets.hand_summary
        trunk_in_dim = encode.trunk_input_dim(
            state_dim,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_summary_in_state=hand_summary_in_state,
            hand_embed_dim=arch.hand_embed_dim,
            pooled_hand_width=arch.pooled_hand_width,
            tray_set_embedding=arch.tray_set_embedding,
            n_playable_multihots=n_extra,
        )
        self.state_trunk, _ = mlp.build_body(
            trunk_in_dim,
            arch.trunk_layers,
            between_activation=arch.trunk_between_activation_resolved,
            final_activation=arch.trunk_final_activation_resolved,
            dropout=arch.trunk_dropout_resolved,
            layernorm=arch.trunk_layernorm_resolved,
        )

    def _build_choice_encoder(
        self, choice_dim: int, arch: architecture.ModelArchitecture
    ) -> None:
        """Register ``choice_encoder``.

        The per-choice encoder reads each candidate's non-identity features plus
        its card identity embedded through the shared card table. Applies a final
        activation when ``arch.encoder_final_activation`` is True."""
        cho = self._choice_embed_offsets()
        choice_in_dim = encode.choice_input_dim(
            choice_dim,
            arch.card_embed_dim,
            include_setup=self.include_setup,
            has_becomes_playable=(cho.becomes_playable is not None),
        )
        self.choice_encoder, _ = mlp.build_body(
            choice_in_dim,
            arch.choice_layers,
            between_activation=arch.choice_between_activation_resolved,
            final_activation=arch.choice_final_activation_resolved,
            dropout=arch.choice_dropout_resolved,
            layernorm=arch.choice_layernorm_resolved,
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
                between_activation=arch.head_between_activation_resolved,
                final_activation=arch.head_final_activation_resolved,
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
            between_activation=arch.value_between_activation_resolved,
            final_activation=arch.value_final_activation_resolved,
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

    def _state_embed_offsets(self) -> StateEmbedOffsets:
        """The column offsets :meth:`_embed_state` splits the flat state vector on
        — the card-index block, hand multi-hot, decision-type tail, and the
        hand-summary slice (used when ``use_distinct_hand_model`` is on).

        The live net reads them from the current ``encode.layout`` chain. In
        v0.9+ the hand-summary stripe is absent from the live vector, so
        ``hand_summary`` and ``hand_summary_end`` are both 0 (a sentinel meaning
        "derive in-model"). Era compat nets carry their own frozen-geometry state
        vector, so they override this to return the offsets that vector was
        written with — never the live ones. Slicing an old vector at live offsets
        is silent corruption: the widths can coincide (no crash) while the columns
        read are wrong, so this seam — *every* offset, not just ``encode_state`` —
        must move with the era (see ``docs/VERSIONING.md`` and
        ``compat/INDEX.md``)."""
        return StateEmbedOffsets(
            card_index=encode.OFF_CARD_INDEX,
            hand_multihot=encode.OFF_HAND_MULTIHOT,
            decision_type=encode.OFF_DECISION_TYPE,
            hand_summary=0,  # removed in v0.9; derived in-model
            hand_summary_end=0,
        )

    def _choice_embed_offsets(self) -> ChoiceEmbedOffsets:
        """The column offsets :meth:`_embed_choices` splits the flat choice row on.

        Returns offsets for the live encoding. Compat subclasses override to return
        their era's frozen offsets (``becomes_playable=None`` for pre-0.6 eras)."""
        return ChoiceEmbedOffsets(
            board_idx=encode.CHOICE_BOARD_IDX_OFFSET,
            bird_id=encode.CHOICE_BIRD_ID_OFFSET,
            becomes_playable=encode.CHOICE_BECOMES_PLAYABLE_OFFSET,
            kept_multihot=(
                encode.CHOICE_KEPT_MULTIHOT_OFFSET if self.include_setup else None
            ),
        )

    def _embed_state(
        self, state: torch.Tensor, card_table: torch.Tensor
    ) -> torch.Tensor:
        """Turn the flat state ``(B, state_dim)`` into the trunk's input by
        replacing the card-identity columns with shared card vectors: the index
        block becomes one ``card_table`` row per slot (flattened), and the hand
        multi-hot (plus any playability multi-hots) becomes hand/set embeddings,
        all concatenated with the continuous features.

        When ``use_distinct_hand_model`` is on the 10-dim hand-summary stripe is
        removed from the continuous block and redirected into the hand encoder,
        so the trunk's continuous feed is correspondingly narrower. Each extra
        playability multi-hot block is embedded through the same encoder (or
        mean-pooled) and appended after the hand embedding. When
        ``tray_set_embedding`` is also on, one tray-*set* embedding is appended.

        When ``use_board_attention`` is on the work is delegated to
        :meth:`_embed_state_board_attention` which runs self-attention over each
        player's 15 board slots before flattening — the total width is identical
        to the non-attention path, so ``trunk_input_dim`` is unchanged."""
        offsets = self._state_embed_offsets()
        if self.arch.use_board_attention:
            return self._embed_state_board_attention(state, card_table, offsets)
        off_index = offsets.card_index
        off_hand = offsets.hand_multihot
        off_decision = offsets.decision_type

        # Card-index block -> per-slot card-table lookups, flattened.
        card_idx = (
            state[:, off_index:off_hand].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
        )
        slot_emb = card_table[card_idx].reshape(card_idx.shape[0], -1)

        # The hand span may contain N extra playability multi-hots appended after
        # the hand multi-hot; n_total encodes how many 180-dim blocks are present.
        hand_span = state[:, off_hand:off_decision]
        n_total = hand_span.shape[-1] // encode.HAND_MULTIHOT_DIM
        hand_multihot = hand_span[:, : encode.HAND_MULTIHOT_DIM]
        extra_multihots = [
            hand_span[
                :, (k * encode.HAND_MULTIHOT_DIM) : ((k + 1) * encode.HAND_MULTIHOT_DIM)
            ]
            for k in range(1, n_total)
        ]

        if self.arch.use_distinct_hand_model:
            hand_sum_off = offsets.hand_summary
            hand_sum_end = offsets.hand_summary_end
            prefix = state[:, :off_index]
            if hand_sum_end > hand_sum_off:
                # Pre-0.9 frozen vector: stripe is physically present — excise it
                # from the continuous prefix and read it for the hand encoder.
                continuous = torch.cat(
                    [
                        prefix[:, :hand_sum_off],
                        prefix[:, hand_sum_end:],
                        state[:, off_decision:],
                    ],
                    dim=-1,
                )
                hand_summary = state[:, hand_sum_off:hand_sum_end]
            else:
                # Live v0.9+: stripe not in vector — continuous prefix is unchanged
                # and the summary is derived in-model from the hand multi-hot.
                continuous = torch.cat([prefix, state[:, off_decision:]], dim=-1)
                hand_summary = hand_model.set_summary_from_multihot(
                    hand_multihot, self.card_summary_matrix[1:]
                )
            hand_emb = hand_model.embed_card_set(
                self.hand_encoder, hand_multihot, hand_summary
            )
            # Embed each extra multi-hot as a card set through the same hand encoder.
            extra_embs = [
                hand_model.embed_card_set(
                    self.hand_encoder,
                    mh,
                    hand_model.set_summary_from_multihot(
                        mh, self.card_summary_matrix[1:]
                    ),
                )
                for mh in extra_multihots
            ]
        else:
            continuous = torch.cat(
                [state[:, :off_index], state[:, off_decision:]], dim=-1
            )
            # Pool hand multi-hot through shared card table rows (skip padding row).
            hand_emb = hand_model.pool_card_set(
                hand_multihot, card_table[1:], self.arch.hand_pooling
            )
            # Extra multi-hots: same pooling mode as the hand.
            extra_embs: list[torch.Tensor] = [
                hand_model.pool_card_set(mh, card_table[1:], self.arch.hand_pooling)
                for mh in extra_multihots
            ]

        if not self.arch.tray_set_embedding:
            return torch.cat([continuous, slot_emb, hand_emb, *extra_embs], dim=-1)

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
        return torch.cat(
            [continuous, slot_emb, tray_set_emb, hand_emb, *extra_embs], dim=-1
        )

    def _embed_state_board_attention(
        self,
        state: torch.Tensor,
        card_table: torch.Tensor,
        offsets: StateEmbedOffsets,
    ) -> torch.Tensor:
        """Board-attention branch of ``_embed_state``.

        Constructs 15-token sequences for each player's board
        (token = card_embed ⊕ 9 mutable scalars), applies masked self-attention
        with a residual, then concatenates the flattened outputs in place of
        the standard per-slot card lookups + board-stripe continuous dims. The
        total output width is identical to the non-attention path — the two
        board continuous stripes (270 dims) are excised from ``continuous`` and
        re-folded into the flattened tokens (15×(E+9) each = 270 dims in total
        per board), keeping ``trunk_input_dim`` unchanged."""
        off_index = offsets.card_index
        off_hand = offsets.hand_multihot
        off_decision = offsets.decision_type

        # Card-index block: [B, 33] — own 0..14, opp 15..29, tray 30..32.
        card_idx = (
            state[:, off_index:off_hand].long().clamp_(0, encode.HAND_MULTIHOT_DIM)
        )

        # Per-slot card table lookups — [B, 33, E], NOT flattened yet.
        slot_emb_all = card_table[card_idx]
        own_card_emb = slot_emb_all[:, : encode.SLOTS_PER_BOARD]
        opp_card_emb = slot_emb_all[
            :, encode.SLOTS_PER_BOARD : encode.N_BOARD_INDEX_SLOTS
        ]
        tray_flat = slot_emb_all[:, encode.N_BOARD_INDEX_SLOTS :].reshape(
            state.shape[0], -1
        )

        # Mutable board scalars [B, 15, 9] for each player.
        off_bme = encode.OFF_BOARD_ME
        off_bopp = encode.OFF_BOARD_OPP
        board_end = off_bopp + encode.BOARD_CONT_STRIPE_DIM
        own_scalars = state[
            :, off_bme : off_bme + encode.BOARD_CONT_STRIPE_DIM
        ].reshape(state.shape[0], encode.SLOTS_PER_BOARD, encode.SLOT_SCALAR_DIM)
        opp_scalars = state[
            :, off_bopp : off_bopp + encode.BOARD_CONT_STRIPE_DIM
        ].reshape(state.shape[0], encode.SLOTS_PER_BOARD, encode.SLOT_SCALAR_DIM)

        # Tokens = [card_embed ⊕ scalars]: [B, 15, E+9].
        own_tokens = torch.cat([own_card_emb, own_scalars], dim=-1)
        opp_tokens = torch.cat([opp_card_emb, opp_scalars], dim=-1)

        # Empty-slot masks: True = padding slot (card_idx == 0).
        own_empty = card_idx[:, : encode.SLOTS_PER_BOARD] == 0
        opp_empty = (
            card_idx[:, encode.SLOTS_PER_BOARD : encode.N_BOARD_INDEX_SLOTS] == 0
        )

        # Apply attention and residual; flatten to [B, 15*(E+9)].
        own_flat = _apply_board_attention(self.board_attn_me, own_tokens, own_empty)
        opp_flat = _apply_board_attention(self.board_attn_opp, opp_tokens, opp_empty)

        # Build continuous prefix with board stripes excised (they rode inside
        # the attention tokens). Boards are adjacent in the layout so one
        # contiguous slice covers both.
        hand_span = state[:, off_hand:off_decision]
        n_total = hand_span.shape[-1] // encode.HAND_MULTIHOT_DIM
        hand_multihot = hand_span[:, : encode.HAND_MULTIHOT_DIM]
        extra_multihots = [
            hand_span[
                :, (k * encode.HAND_MULTIHOT_DIM) : ((k + 1) * encode.HAND_MULTIHOT_DIM)
            ]
            for k in range(1, n_total)
        ]

        if self.arch.use_distinct_hand_model:
            hand_sum_off = offsets.hand_summary
            hand_sum_end = offsets.hand_summary_end
            if hand_sum_end > hand_sum_off:
                # Pre-0.9 frozen vector: excise board_me + board_opp AND
                # hand_summary from the prefix; read summary from state.
                continuous = torch.cat(
                    [
                        state[:, :off_bme],
                        state[:, board_end:hand_sum_off],
                        state[:, hand_sum_end:off_index],
                        state[:, off_decision:],
                    ],
                    dim=-1,
                )
                hand_summary = state[:, hand_sum_off:hand_sum_end]
            else:
                # Live v0.9+: excise only board_me + board_opp; derive summary
                # in-model from the hand multi-hot.
                continuous = torch.cat(
                    [
                        state[:, :off_bme],
                        state[:, board_end:off_index],
                        state[:, off_decision:],
                    ],
                    dim=-1,
                )
                hand_summary = hand_model.set_summary_from_multihot(
                    hand_multihot, self.card_summary_matrix[1:]
                )
            hand_emb = hand_model.embed_card_set(
                self.hand_encoder, hand_multihot, hand_summary
            )
            extra_embs = [
                hand_model.embed_card_set(
                    self.hand_encoder,
                    mh,
                    hand_model.set_summary_from_multihot(
                        mh, self.card_summary_matrix[1:]
                    ),
                )
                for mh in extra_multihots
            ]
        else:
            # Excise board_me + board_opp from the prefix only.
            continuous = torch.cat(
                [
                    state[:, :off_bme],
                    state[:, board_end:off_index],
                    state[:, off_decision:],
                ],
                dim=-1,
            )
            # Pool hand multi-hot through shared card table rows (skip padding row).
            hand_emb = hand_model.pool_card_set(
                hand_multihot, card_table[1:], self.arch.hand_pooling
            )
            extra_embs: list[torch.Tensor] = [
                hand_model.pool_card_set(mh, card_table[1:], self.arch.hand_pooling)
                for mh in extra_multihots
            ]

        # Tray: identical to the standard path (tray_flat already computed).
        if not self.arch.tray_set_embedding:
            return torch.cat(
                [continuous, own_flat, opp_flat, tray_flat, hand_emb, *extra_embs],
                dim=-1,
            )

        # Tray-set embedding through the shared hand encoder.
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
        return torch.cat(
            [
                continuous,
                own_flat,
                opp_flat,
                tray_flat,
                tray_set_emb,
                hand_emb,
                *extra_embs,
            ],
            dim=-1,
        )

    def _embed_choices(
        self, choices: torch.Tensor, card_table: torch.Tensor
    ) -> torch.Tensor:
        """Turn the per-choice features ``(B, K, choice_dim)`` into the choice
        encoder's input by mapping the candidate's card regions through the
        shared card table and concatenating them with the remaining features.

        The 15-slot board-index block sits immediately before the candidate
        bird-index column; both become shared-embedding lookups. The
        ``becomes_playable`` multi-hot (0.6+) and the trailing ``kept_multihot``
        stripe (when ``include_setup``) are each summed through the card table
        into one more embedding. Everything else passes through. Pre-0.6 compat
        shims return ``becomes_playable=None`` from ``_choice_embed_offsets``
        and take the legacy code path."""
        cho = self._choice_embed_offsets()
        off_board = cho.board_idx
        off_bird = cho.bird_id
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

        off_becomes = cho.becomes_playable  # None for pre-0.6
        off_kept = cho.kept_multihot  # None when not include_setup

        if off_becomes is not None and off_kept is not None:
            # 0.6+, include_setup: becomes_playable then setup_agg then kept_multihot.
            off_setup = off_becomes + encode.CHOICE_BECOMES_PLAYABLE_DIM
            becomes_mh = choices[..., off_becomes:off_setup]
            becomes_emb = becomes_mh @ card_table[1:]
            kept_mh = choices[..., off_kept:]
            kept_emb = kept_mh @ card_table[1:]
            rest = torch.cat(
                [
                    choices[..., :off_board],
                    choices[..., end_bird:off_becomes],
                    choices[..., off_setup:off_kept],
                ],
                dim=-1,
            )
            return torch.cat([rest, cand_emb, board_emb, becomes_emb, kept_emb], dim=-1)
        elif off_becomes is not None:
            # 0.6+, no setup: becomes_playable is the last stripe.
            becomes_mh = choices[..., off_becomes:]
            becomes_emb = becomes_mh @ card_table[1:]
            rest = torch.cat(
                [choices[..., :off_board], choices[..., end_bird:off_becomes]], dim=-1
            )
            return torch.cat([rest, cand_emb, board_emb, becomes_emb], dim=-1)
        elif off_kept is not None:
            # Pre-0.6, include_setup: no becomes_playable stripe.
            kept_mh = choices[..., off_kept:]
            kept_emb = kept_mh @ card_table[1:]
            rest = torch.cat(
                [choices[..., :off_board], choices[..., end_bird:off_kept]], dim=-1
            )
            return torch.cat([rest, cand_emb, board_emb, kept_emb], dim=-1)
        else:
            # Pre-0.6, no setup: simplest case.
            rest = torch.cat(
                [choices[..., :off_board], choices[..., end_bird:]], dim=-1
            )
            return torch.cat([rest, cand_emb, board_emb], dim=-1)


###### MODULE-LEVEL HELPERS ######


def _apply_board_attention(
    attn: nn.MultiheadAttention,
    tokens: torch.Tensor,
    empty: torch.Tensor,
) -> torch.Tensor:
    """Masked self-attention with residual over one player's 15 board slots.

    Args:
        attn:   the ``nn.MultiheadAttention`` module for this player's board.
        tokens: ``(B, 15, E+9)`` — one token per slot (card embed ⊕ scalars).
                Empty slots have zero-vector tokens (``card_pad_mask`` + zero scalars).
        empty:  ``(B, 15)`` bool, True = empty slot (card_idx == 0).

    Returns:
        ``(B, 15*(E+9))`` — attended+residual tokens, flattened.

    The NaN guard: when ALL 15 slots in a row are empty (common at game start),
    ``key_padding_mask=empty`` would make every key masked, causing ``softmax``
    over ``-inf`` → NaN.  We clone the mask and unmask slot 0 as a dummy key for
    those rows. Because the dummy token is a zero vector the attention output for
    those rows is ≈ 0; the subsequent ``masked_fill`` and the zero token ensure
    the residual is exactly 0 for every empty slot either way."""
    # Guard: unmask slot 0 as dummy key for fully-empty boards.
    safe = empty.clone()
    safe[empty.all(1), 0] = False

    out, _ = attn(tokens, tokens, tokens, key_padding_mask=safe, need_weights=False)

    # Zero contributions from empty query slots (they attended the dummy key).
    out = out.masked_fill(empty.unsqueeze(-1), 0.0)

    # Residual: empty rows → 0 + 0 = 0; filled rows → token + context.
    return (tokens + out).reshape(tokens.shape[0], -1)
