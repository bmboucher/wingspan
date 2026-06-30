"""The setup model network: a two-tower actor-critic over a setup candidate,
mirroring the in-game ``model.PolicyValueNet`` and reading the main net's card
representations.

``SetupNet`` consumes one
:func:`wingspan.setup_model.encode.encode_setup_candidate` feature vector through
two trunks and two heads, exactly paralleling the in-game net:

* a **state trunk** (``trunk_layers``) encodes the action-independent stripes
  (``_embed_state``: the tray as per-slot card-table rows, the raw birdfeeder /
  round-goal passthrough, and the bonus-cards-on-offer multi-hot in ``split_bonus``
  mode) into a shared ``state_enc``;
* a **choice trunk** (``choice_layers``) encodes the action stripes
  (``_embed_choice``: the kept-cards / playability sets, kept foods, the bonus
  action, the kept-bonus pricing and goal affinity) into ``choice_enc``;
* the **value head** (``forward`` / the critic ``V(s)``) reads ``state_enc`` only,
  so its output is identical for every keep candidate of a deal: a true value
  baseline, not the post-keep ``Q(s, a)``. This is what makes the setup advantage
  ``target − V(s)`` carry a real gradient (``docs/TRAINING.md §6.5``);
* the **policy head** (``policy_logits`` / selection + REINFORCE actor) reads
  ``cat(state_enc, choice_enc)``, so its logits rank candidate keeps. Candidate
  selection at collection time uses these logits.

The shared state trunk feeds both heads, so the value and policy heads share a
learned state representation (the point of the two-tower design). Because ``V(s)``
is a pure function of state, candidate ranking REQUIRES the policy head
(``arch.use_policy_head=True``, always set in practice); a value-only configuration
would score every candidate identically. ``policy_and_value`` returns
``(policy_logits (B,), value_preds (B,))`` for inspection, computing ``state_enc``
once for both heads; the learner computes ``V(s)`` once per deal via ``forward``.

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
setup trunks/heads score keeps in the representation most meaningful to the in-game
model. The frozen inputs shift as the main net trains — the accepted trade-off; the
on-policy updates already track moving targets each iteration.
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
    """A two-tower actor-critic scoring one setup candidate.

    See the module docstring for the state-trunk / choice-trunk split and the
    value-only (``use_policy_head=False``) vs actor-critic distinction.
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

        # ---- State trunk: encodes the action-independent stripes
        # (``setup_state_input_dim``) into a shared ``state_enc`` feeding both
        # heads — the in-game net's ``state_trunk``. The critic is therefore a
        # function of the deal state alone (``V(s)``, not ``Q(s, a)``), so the
        # setup advantage ``target − V(s)`` does not self-cancel
        # (``docs/TRAINING.md §6.5``). No LayerNorm; the trunk uses
        # ``between_activation`` as its own final activation so ``state_enc`` is
        # nonlinear before the heads' first Linear. ----
        state_in = setup_model.setup_state_input_dim(encoding, main_arch)
        self.state_trunk, state_enc_dim = mlp.build_body(
            state_in,
            arch.trunk_layers,
            between_activation=arch.between_activation,
            final_activation=arch.between_activation,
            dropout=arch.dropout,
            layernorm=False,
        )

        # ---- Choice trunk: encodes the action stripes
        # (``setup_choice_input_dim``) into ``choice_enc`` — the in-game net's
        # ``choice_encoder``. Present only with a policy head (value-only mode has
        # neither choice trunk nor policy head). ----
        if arch.use_policy_head:
            choice_in = setup_model.setup_choice_input_dim(encoding, main_arch)
            choice_trunk_seq, choice_enc_dim = mlp.build_body(
                choice_in,
                arch.choice_layers,
                between_activation=arch.between_activation,
                final_activation=arch.between_activation,
                dropout=arch.dropout,
                layernorm=False,
            )
            self.choice_trunk: nn.Module = choice_trunk_seq
        else:
            self.choice_trunk = nn.Identity()
            choice_enc_dim = 0

        # Policy head: logits over candidate keeps for selection + REINFORCE,
        # reading ``cat(state_enc, choice_enc)``. Present only when
        # ``arch.use_policy_head`` is True (always, in practice).
        self.policy_head: nn.Sequential | None = (
            mlp.build_readout(
                state_enc_dim + choice_enc_dim,
                arch.head_layers,
                between_activation=arch.between_activation,
                final_activation=arch.final_activation,
                dropout=arch.dropout,
            )
            if arch.use_policy_head
            else None
        )

        # Value head: the trainable readout MLP predicting the normalized return
        # ``V(s)`` (in ``score_norm`` units) from the shared ``state_enc``.
        self.value_head = mlp.build_readout(
            state_enc_dim,
            arch.value_layers,
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
        (``_embed_state`` → the shared ``state_trunk``), so the prediction is
        identical for every keep candidate of a deal: a true value baseline, not
        ``Q(s, a)``."""
        return self.value_head(self._state_enc(features)).squeeze(-1)

    def policy_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Per-candidate selection logits via the policy head: ``(B, feature_dim)`` -> ``(B,)``.

        Reads ``cat(state_enc, choice_enc)`` — the shared state encoding plus the
        choice trunk's encoding of the action stripes (``_embed_choice``). Requires
        ``arch.use_policy_head=True``; raises ``RuntimeError`` otherwise."""
        if self.policy_head is None:
            raise RuntimeError(
                "policy_logits called on a SetupNet without a policy head "
                "(arch.use_policy_head=False)"
            )
        state_enc = self._state_enc(features)
        choice_enc = self.choice_trunk(self._embed_choice(features))
        return self.policy_head(torch.cat([state_enc, choice_enc], dim=-1)).squeeze(-1)

    def policy_and_value(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch via both heads: ``(B, feature_dim)`` -> ``(policy_logits (B,), value_preds (B,))``.

        Computes the shared ``state_enc`` once: the value head reads it directly
        (so ``value_preds`` is constant across the candidates of one deal — it is
        ``V(s)``, ignoring the action), and the policy head reads
        ``cat(state_enc, choice_enc)``. Requires ``arch.use_policy_head=True``;
        raises ``RuntimeError`` otherwise. The learner computes ``V(s)`` once per
        deal via ``forward`` rather than over every candidate; this convenience
        pairing is for inspection/tests."""
        if self.policy_head is None:
            raise RuntimeError(
                "policy_and_value called on a SetupNet without a policy head "
                "(arch.use_policy_head=False)"
            )
        state_enc = self._state_enc(features)
        choice_enc = self.choice_trunk(self._embed_choice(features))
        policy = self.policy_head(torch.cat([state_enc, choice_enc], dim=-1)).squeeze(
            -1
        )
        value = self.value_head(state_enc).squeeze(-1)
        return policy, value

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

    def _state_enc(self, features: torch.Tensor) -> torch.Tensor:
        """The shared state encoding: the state trunk over the action-independent
        stripes. Feeds the value head directly and the policy head concatenated
        with the choice encoding."""
        return self.state_trunk(self._embed_state(features))

    def _embed_choice(self, features: torch.Tensor) -> torch.Tensor:
        """The action-only readout input for the choice trunk (the policy path).

        The mirror image of ``_embed_state``: it gathers *only* the keep-dependent
        stripes — the kept-cards set (embedded as one set vector), kept foods, the
        bonus action (the folded ``kept_bonus`` one-hot, or the
        ``bonus_card_affinity`` min/max in ``split_bonus`` mode), ``kept_bonus_value``
        (folded mode only), the per-round ``goal_affinity``, and each appended
        playability multi-hot (``turn1_playable`` / ``playable_kept_cards``) as one
        extra set vector.

        Like ``_embed_state`` this is a stripe-aware gather by ``SetupEncoding``
        offsets; the empty-slice arithmetic lets one body handle folded /
        ``split_bonus`` / ``split_food`` alike. Together the two gathers partition
        the embedded candidate vector (``setup_choice_input_dim`` +
        ``setup_state_input_dim``)."""
        card_table = self._card_table_for_pass()  # (181, M)
        enc = self.encoding

        # Kept-cards set -> one set embedding matching the main net's path.
        kept_emb = self._embed_card_set(features[..., : enc.kept_cards_dim], card_table)

        # Raw action passthroughs (empty-slice when a flag removes the stripe):
        #   foods        — kept-foods multi-hot (omitted in split_food mode)
        #   bonus_action — folded kept_bonus one-hot, or split bonus_card_affinity
        #                  (skips the leading on-offer bonus_cards STATE stripe)
        #   bonus_value  — kept_bonus_value pricing (folded mode only)
        #   goal_affinity — one scalar per round
        foods = features[..., enc.kept_cards_dim : enc.off_bonus_block]
        bonus_action = features[
            ..., enc.off_bonus_block + enc.bonus_cards_dim : enc.off_tray
        ]
        bonus_value = features[..., enc.off_bonus_value : enc.off_goal_affinity]
        goal_affinity = features[..., enc.off_goal_affinity : enc.off_turn1_playable]

        # Embed any appended 180-dim card-set multi-hots (turn1_playable, then
        # playable_kept_cards), each as one extra set embedding.
        appended: list[torch.Tensor] = []
        for stripe_off in self._appended_multihot_offsets():
            multihot = features[..., stripe_off : stripe_off + enc.kept_cards_dim]
            appended.append(self._embed_card_set(multihot, card_table))

        return torch.cat(
            [kept_emb, foods, bonus_action, bonus_value, goal_affinity] + appended,
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
