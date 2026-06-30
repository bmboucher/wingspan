"""The setup model network: an actor-critic over a setup candidate, reading the
main net's card representations.

``SetupNet`` consumes one
:func:`wingspan.setup_model.encode.encode_setup_candidate` feature vector
through two heads with **different inputs**:

* **Policy head** (``policy_logits`` / selection + REINFORCE actor): reads the
  FUSED state ⊕ action embedding — the full per-candidate vector — so its logits
  rank candidate keeps. Candidate selection at collection time uses these logits.
* **Value head** (``forward`` / the critic ``V(s)``): reads a STATE-ONLY
  embedding — only the action-independent stripes (tray, birdfeeder, round goals,
  and the bonus-cards-on-offer multi-hot in ``split_bonus`` mode). Its output is
  therefore identical for every keep candidate of a deal: a true value baseline,
  not the post-keep ``Q(s, a)``. This is what makes the setup advantage
  ``target − V(s)`` carry a real gradient (``docs/TRAINING.md §6.5``);
  ``policy_and_value`` returns ``(policy_logits (B,), value_preds (B,))`` for
  inspection, but the learner computes ``V(s)`` once per deal via ``forward``.

Because ``V(s)`` ignores the action, candidate ranking REQUIRES the policy head
(``arch.use_policy_head=True``, always set in practice); a value-only
configuration would score every candidate identically.

The card-identity blocks are embedded in-net through copies of the main net's
two shared embedders, built from the same :func:`wingspan.model.mlp.build_body` recipe
so ``load_state_dict`` syncs them weight-for-weight:

* the **card encoder** (single-card ``M``-dim table) embeds the three tray index
  columns, one card-table row per slot — always frozen and re-synced from the
  main net each iteration (the main net has a card encoder unconditionally);
* the **hand encoder** (multi-card set embedder) is used **only** when the main
  architecture has ``use_distinct_hand_model=True``: it embeds card sets via
  ``[multi-hot ⊕ derived 10-dim set summary]``, frozen + synced each iteration.
  When ``use_distinct_hand_model=False`` (the default), card sets are embedded
  by permutation-invariant pooling over the synced card table
  (``hand_model.pool_card_set``), identical to the main net's hand-pooling path.
  The hand encoder module is built in all cases so ``load_state_dict`` transfers
  its weights when the main net has one.

Setup is the most data-starved decision (~2 samples/game) while the embedders
learn from hundreds of in-game decisions per game; freezing the copies lets the
setup MLP (``arch.hidden_layers`` → 1, built by ``mlp.build_readout``) score
keeps in the representation most meaningful to the in-game model. The frozen
inputs shift as the main net trains — the accepted trade-off; the on-policy
updates already track moving targets each iteration.
"""

from __future__ import annotations

import typing

import torch
from torch import nn

from wingspan import architecture, encode, setup_model
from wingspan.model import hand_model, mlp

if typing.TYPE_CHECKING:
    from wingspan.training import setup_runmeta


class SetupNet(nn.Module):
    """A net scoring one setup candidate; optionally with a separate policy head.

    See module docstring for the value-only vs actor-critic distinction.
    """

    # Constant buffers registered in __init__ (declared here so the type checker
    # sees them as tensors rather than nn.Module's generic attribute access).
    card_features: torch.Tensor
    card_pad_mask: torch.Tensor
    card_summary_matrix: torch.Tensor

    def __init__(
        self,
        *,
        encoding: setup_model.SetupEncoding | None = None,
        arch: setup_model.SetupArchitecture | None = None,
        main_arch: architecture.ModelArchitecture | None = None,
    ):
        super().__init__()
        # Lazily-filled cache of the inference card table (the setup twin of
        # ``model.PolicyValueNet._inference_card_table``). ``None`` whenever the
        # synced weights or the train/eval mode may have changed; recomputed on
        # the next eval forward. Set before any buffer registration.
        self._inference_card_table: torch.Tensor | None = None
        if encoding is None:
            encoding = setup_model.SetupEncoding()
        if arch is None:
            arch = setup_model.SetupArchitecture()
        if main_arch is None:
            main_arch = architecture.ModelArchitecture()
        self.encoding = encoding
        self.feature_dim = encoding.total_dim
        self.arch = arch
        self.main_arch = main_arch

        # The frozen single-card embedder copy: recipe identical to the main
        # net's card encoder so its state_dict syncs exactly. Frozen always —
        # the main net trains it; this net only reads its table.
        # Delegated to _build_card_encoder so compat subclasses can pin a
        # previous CARD_FEATURE_DIM without touching any other block.
        self._build_card_encoder(main_arch)

        # The multi-card set embedder copy (the main net's hand encoder). Frozen
        # + synced when the main architecture actually has one; otherwise this is
        # the setup net's own trainable set encoder (there is nothing to sync
        # from), keeping main configs without the distinct hand model valid.
        self._hand_encoder_frozen = main_arch.use_distinct_hand_model
        self.hand_encoder, _ = mlp.build_body(
            encode.HAND_ENCODER_INPUT_DIM,
            main_arch.hand_encoder_layers + (main_arch.hand_embed_width,),
            between_activation=main_arch.hand_between_activation_resolved,
            final_activation=main_arch.hand_final_activation_resolved,
            dropout=main_arch.hand_dropout_resolved,
            layernorm=main_arch.hand_layernorm_resolved,
        )
        if self._hand_encoder_frozen:
            self.hand_encoder.requires_grad_(False)

        # Constant card tables (``persistent=False`` keeps them out of the
        # checkpoint / broadcast payloads — they rebuild from the catalog).
        self.register_buffer(
            "card_summary_matrix",
            torch.tensor(encode.card_summary_matrix(), dtype=torch.float32),
            persistent=False,
        )

        # ---- Policy path: optional trunk + policy head over the FUSED
        # (state ⊕ action) per-candidate embedding. Empty trunk_layers →
        # nn.Identity (no state_dict keys, so a trunk-less checkpoint loads
        # cleanly via load_state_dict). ----
        readout_in = setup_model.setup_readout_input_dim(
            encoding.total_dim,
            main_arch,
            include_turn1_playable=encoding.include_turn1_playable,
            include_playable_kept_cards=encoding.include_playable_kept_cards,
        )
        policy_head_in = readout_in
        if arch.trunk_layers:
            trunk_seq, policy_head_in = mlp.build_body(
                readout_in,
                arch.trunk_layers,
                # Trunk uses between_activation as its final activation so the
                # trunk output is nonlinear before the head's first Linear.
                between_activation=arch.between_activation,
                final_activation=arch.between_activation,
                dropout=arch.dropout,
                layernorm=False,
            )
            self.trunk: nn.Module = trunk_seq
        else:
            self.trunk = nn.Identity()

        # Policy head: logits over candidate keeps for selection + REINFORCE.
        # Present only when ``arch.use_policy_head`` is True (always, in practice).
        self.policy_mlp: nn.Sequential | None = (
            mlp.build_readout(
                policy_head_in,
                arch.hidden_layers,
                between_activation=arch.between_activation,
                final_activation=arch.final_activation,
                dropout=arch.dropout,
            )
            if arch.use_policy_head
            else None
        )

        # ---- Value path: optional trunk + value head over the STATE-ONLY
        # embedding (``setup_state_input_dim``). The critic is a function of the
        # deal state alone — ``V(s)``, not ``Q(s, a)`` — so the setup advantage
        # ``target − V(s)`` no longer self-cancels (``docs/TRAINING.md §6.5``). ----
        state_in = setup_model.setup_state_input_dim(encoding, main_arch)
        value_head_in = state_in
        if arch.value_trunk_layers:
            value_trunk_seq, value_head_in = mlp.build_body(
                state_in,
                arch.value_trunk_layers,
                between_activation=arch.between_activation,
                final_activation=arch.between_activation,
                dropout=arch.dropout,
                layernorm=False,
            )
            self.value_trunk: nn.Module = value_trunk_seq
        else:
            self.value_trunk = nn.Identity()

        # Value head: the trainable readout MLP predicting the normalized return
        # ``V(s)`` (in ``score_norm`` units) from the state-only embedding.
        self.mlp = mlp.build_readout(
            value_head_in,
            arch.value_hidden_resolved,
            between_activation=arch.between_activation,
            final_activation=arch.final_activation,
            dropout=arch.dropout,
        )

    def _build_card_encoder(self, main_arch: architecture.ModelArchitecture) -> None:
        """Build the frozen card encoder and register ``card_features`` / ``card_pad_mask``.

        Overridden by compat subclasses (e.g. ``compat.v0_1.SetupNetV01``) to
        pin the card-encoder input width and feature table to a prior artifact
        era's geometry without touching any other block."""
        self.card_encoder, _ = mlp.build_body(
            encode.CARD_FEATURE_DIM,
            main_arch.card_encoder_layers + (main_arch.card_embed_dim,),
            between_activation=main_arch.card_between_activation_resolved,
            final_activation=main_arch.card_final_activation_resolved,
            dropout=main_arch.card_dropout_resolved,
            layernorm=main_arch.card_layernorm_resolved,
        )
        self.card_encoder.requires_grad_(False)
        self.register_buffer(
            "card_features",
            torch.tensor(encode.card_feature_matrix(), dtype=torch.float32),
            persistent=False,
        )
        pad_mask = torch.ones(encode.HAND_MULTIHOT_DIM + 1, 1)
        pad_mask[0] = 0.0
        self.register_buffer("card_pad_mask", pad_mask, persistent=False)

    @classmethod
    def from_setup_config(cls, descriptor: "setup_runmeta.SetupConfig") -> "SetupNet":
        """Rebuild a net matching a saved ``setup_config.json`` descriptor — fresh
        weights in the saved shape, ready for ``load_state_dict``."""
        return cls(
            encoding=descriptor.setup_encoding,
            arch=descriptor.setup_arch,
            main_arch=descriptor.main_arch,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """The state-only critic ``V(s)``: ``(B, feature_dim)`` -> ``(B,)``.

        Reads *only* the action-independent stripes of each candidate vector
        (the tray as ``TRAY_SIZE`` per-slot card-table rows, plus the raw
        birdfeeder / round-goal passthrough and — in ``split_bonus`` mode — the
        bonus-cards-on-offer multi-hot), so the prediction is identical for every
        keep candidate of a deal: a true value baseline, not ``Q(s, a)``."""
        return self.mlp(self.value_trunk(self._embed_state(features))).squeeze(-1)

    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Per-candidate selection logits via the policy head: ``(B, feature_dim)`` -> ``(B,)``.

        Reads the FUSED state ⊕ action embedding (card-set multi-hots become
        pooled/encoder set vectors; the tray becomes per-slot card-table rows;
        all other blocks pass through). Requires ``arch.use_policy_head=True``;
        raises ``RuntimeError`` otherwise."""
        if self.policy_mlp is None:
            raise RuntimeError(
                "policy_logits called on a SetupNet without a policy head "
                "(arch.use_policy_head=False)"
            )
        return self.policy_mlp(self.trunk(self._embed(features))).squeeze(-1)

    def policy_and_value(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch via both heads: ``(B, feature_dim)`` -> ``(policy_logits (B,), value_preds (B,))``.

        The policy head reads the fused state ⊕ action embedding; the value head
        reads the state-only embedding, so ``value_preds`` is constant across the
        candidates of one deal (it is ``V(s)``, ignoring the action). Requires
        ``arch.use_policy_head=True``; raises ``RuntimeError`` otherwise. The
        learner computes ``V(s)`` once per deal via ``forward`` rather than over
        every candidate; this convenience pairing is for inspection/tests."""
        return self.policy_logits(features), self.forward(features)

    def card_table(self) -> torch.Tensor:
        """The frozen ``[181, M]`` card table: the constant card-feature matrix
        mapped through the synced card-encoder copy, with the padding row (index
        0) forced to zero so an empty tray slot contributes a zero vector —
        identical math to ``model.PolicyValueNet.card_table``."""
        return self.card_encoder(self.card_features) * self.card_pad_mask

    def train(self, mode: bool = True) -> "SetupNet":
        """Flip train/eval mode, invalidating the cached inference card table and
        re-pinning the frozen embedder copies to eval.

        Every way the synced weights can change is bracketed by a mode flip
        through here: the loop's per-iteration sync helper ends with ``eval()``,
        and workers ``eval()`` after every broadcast reload — so a stale table is
        never served. Pinning the frozen submodules to eval keeps a nonzero main
        ``dropout`` from adding noise to the frozen features during the setup
        MLP's own training steps."""
        self._inference_card_table = None
        super().train(mode)
        self.card_encoder.eval()
        if self._hand_encoder_frozen:
            self.hand_encoder.eval()
        return self

    ###### PRIVATE #######

    def _embed(self, features: torch.Tensor) -> torch.Tensor:
        """Compute the shared readout input from raw ``(B, feature_dim)`` features.

        Replaces every card-set multi-hot (kept cards, appended playability
        stripes) with a pooled or encoder-embedded set vector — matching the main
        net's embedding path for consistency. The tray index columns become
        ``TRAY_SIZE`` per-slot card-table rows only (no tray-set term). All other
        blocks pass through unchanged. Used by both ``forward`` and
        ``policy_and_value`` so the embedding is computed once."""
        card_table = self._card_table_for_pass()  # (181, M)

        # Slice the raw vector using encoding-aware offsets.
        enc = self.encoding
        kept_multihot = features[..., : enc.kept_cards_dim]
        passthrough = features[..., enc.kept_cards_dim : enc.off_tray]
        tray_idx = (
            features[..., enc.off_tray : enc.off_feeder]
            .long()
            .clamp_(0, encode.HAND_MULTIHOT_DIM)
        )
        # The feeder/goals passthrough ends at the first appended multi-hot stripe.
        feeder_goals_rest = features[..., enc.off_feeder : enc.off_turn1_playable]

        # Kept set -> one set embedding matching the main net's path.
        kept_emb = self._embed_card_set(kept_multihot, card_table)

        # Tray -> 3 M-dim card-table rows only (no tray-set embedding).
        tray_slot_emb = card_table[tray_idx].reshape(*tray_idx.shape[:-1], -1)

        # Embed any appended 180-dim card-set multi-hots (turn1_playable, then
        # playable_kept_cards), each as one extra set embedding.
        appended: list[torch.Tensor] = []
        for stripe_off in self._appended_multihot_offsets():
            mh = features[..., stripe_off : stripe_off + enc.kept_cards_dim]
            appended.append(self._embed_card_set(mh, card_table))

        return torch.cat(
            [kept_emb, passthrough, tray_slot_emb, feeder_goals_rest] + appended,
            dim=-1,
        )

    def _embed_state(self, features: torch.Tensor) -> torch.Tensor:
        """The state-only readout input for the value head ``V(s)``.

        Gathers *only* the action-independent stripes — the tray (as
        ``TRAY_SIZE`` per-slot card-table rows), the raw birdfeeder + round-goal
        passthrough, and (in ``split_bonus`` mode) the bonus-cards-on-offer
        multi-hot. These stripes are byte-identical across every keep candidate
        of a deal, so the value head's output is invariant to the chosen action.

        The stripes are NON-CONTIGUOUS in the raw vector (the keep-dependent
        ``kept_bonus_value`` / ``goal_affinity`` sit between goals and the
        appended sets, and the bonus-block affinity sits right after the on-offer
        multi-hot), so this is a stripe-aware gather by ``SetupEncoding`` offsets,
        never a single slice. Order: tray rows ⊕ feeder ⊕ goals ⊕ bonus-on-offer,
        matching ``setup_model.setup_state_input_dim``."""
        card_table = self._card_table_for_pass()  # (181, M)
        enc = self.encoding

        # Tray -> 3 M-dim card-table rows (state; same path as ``_embed``).
        tray_idx = (
            features[..., enc.off_tray : enc.off_feeder]
            .long()
            .clamp_(0, encode.HAND_MULTIHOT_DIM)
        )
        tray_slot_emb = card_table[tray_idx].reshape(*tray_idx.shape[:-1], -1)

        # Birdfeeder + round goals: raw passthrough between off_feeder and the
        # start of the keep-dependent pricing block (off_bonus_value == end of goals).
        feeder = features[..., enc.off_feeder : enc.off_goals]
        goals = features[..., enc.off_goals : enc.off_bonus_value]

        parts = [tray_slot_emb, feeder, goals]
        if enc.split_bonus:
            # Leading ``bonus_cards_dim`` of the bonus block = the on-offer
            # multi-hot (state); the trailing 2 affinity dims are action.
            parts.append(
                features[
                    ..., enc.off_bonus_block : enc.off_bonus_block + enc.bonus_cards_dim
                ]
            )
        return torch.cat(parts, dim=-1)

    def _embed_card_set(
        self, multihot: torch.Tensor, card_table: torch.Tensor
    ) -> torch.Tensor:
        """Embed one 180-dim card-set multi-hot, matching the main net's path.

        Pools over the synced card table's bird rows when the main architecture
        uses ``use_distinct_hand_model=False`` (the default pooling path).
        Falls through to the frozen hand encoder otherwise."""
        if not self.main_arch.use_distinct_hand_model:
            return hand_model.pool_card_set(
                multihot, card_table[1:], self.main_arch.hand_pooling
            )
        summary = hand_model.set_summary_from_multihot(
            multihot, self.card_summary_matrix[1:]
        )
        return hand_model.embed_card_set(self.hand_encoder, multihot, summary)

    def _appended_multihot_offsets(self) -> list[int]:
        """Absolute feature-vector offsets of each appended 180-dim card-set
        multi-hot stripe, in encoding order."""
        enc = self.encoding
        offsets: list[int] = []
        if enc.include_turn1_playable:
            offsets.append(enc.off_turn1_playable)
        if enc.include_playable_kept_cards:
            offsets.append(enc.off_playable_kept_cards)
        return offsets

    def _card_table_for_pass(self) -> torch.Tensor:
        """The card table for one forward pass: recomputed per call in training
        mode, computed once and memoized at inference (the dominant use — scoring
        504 candidates per deal during collection). The cache is dropped on every
        train/eval flip (``train``), mirroring the main net's contract."""
        if self.training:
            return self.card_table()
        cached = self._inference_card_table
        if cached is None:
            cached = self.card_table().detach()
            self._inference_card_table = cached
        return cached
