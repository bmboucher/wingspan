"""The setup model network: a value-regressor (and optional policy head) over a
setup candidate, reading the main net's card representations.

``SetupNet`` consumes one
:func:`wingspan.setup_model.encode.encode_setup_candidate` feature vector and
emits either one or two scalars depending on whether the actor-critic mode is
enabled (``arch.use_policy_head``):

* **Value-only** (default): ``forward(features)`` returns ``(B,)`` predicted
  score margins.  The setup policy is a softmax over these predicted margins for
  all 504 candidate keeps (``setup_model.select_by_margins``).
* **Actor-critic**: ``forward`` still returns value scalars (backward compat);
  ``policy_and_value(features)`` returns ``(policy_logits (B,), value_preds (B,))``.
  Candidate selection uses policy logits; training uses REINFORCE + value MSE.

The card-identity blocks are embedded in-net through copies of the main net's
two shared embedders, built from the same :func:`wingspan.model.mlp.build_body` recipe
so ``load_state_dict`` syncs them weight-for-weight:

* the **card encoder** (single-card ``M``-dim table) embeds the three tray index
  columns, one card-table row per slot — always frozen and re-synced from the
  main net each iteration (the main net has a card encoder unconditionally);
* the **hand encoder** (multi-card ``N``-dim set embedder) embeds the kept-cards
  set and the tray set, each as ``[multi-hot ⊕ derived 10-dim set summary]`` —
  frozen + synced when the main architecture has ``use_distinct_hand_model``,
  otherwise the copy is this net's own trainable block (so default configs
  remain valid).

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

        # Value head: the trainable readout MLP predicting score margin.
        readout_in = setup_model.setup_readout_input_dim(
            encoding.total_dim,
            main_arch,
            include_turn1_playable=encoding.include_turn1_playable,
            include_playable_kept_cards=encoding.include_playable_kept_cards,
        )
        self.mlp = mlp.build_readout(
            readout_in,
            arch.hidden_layers,
            between_activation=arch.between_activation,
            final_activation=arch.final_activation,
            dropout=arch.dropout,
        )

        # Policy head: a second readout MLP of identical architecture whose
        # logits drive candidate selection and REINFORCE training. Present only
        # when ``arch.use_policy_head`` is True (actor-critic mode).
        self.policy_mlp: nn.Sequential | None = (
            mlp.build_readout(
                readout_in,
                arch.hidden_layers,
                between_activation=arch.between_activation,
                final_activation=arch.final_activation,
                dropout=arch.dropout,
            )
            if arch.use_policy_head
            else None
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
        """Score a batch of setup candidates via the value head: ``(B, feature_dim)`` -> ``(B,)``.

        The kept-cards multi-hot becomes one set embedding, the tray index
        columns become per-slot card-table rows plus one tray-set embedding, and
        the remaining blocks (foods, bonus, feeder, goals) pass through to the
        value readout MLP in their encoded order."""
        return self.mlp(self._embed(features)).squeeze(-1)

    def policy_and_value(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a batch via both heads: ``(B, feature_dim)`` -> ``(policy_logits (B,), value_preds (B,))``.

        Requires ``arch.use_policy_head=True``; raises ``RuntimeError`` otherwise.
        Both heads run on the same embedded representation, computed once."""
        if self.policy_mlp is None:
            raise RuntimeError(
                "policy_and_value called on a SetupNet without a policy head "
                "(arch.use_policy_head=False)"
            )
        embedded = self._embed(features)
        policy_logits = self.policy_mlp(embedded).squeeze(-1)
        value_preds = self.mlp(embedded).squeeze(-1)
        return policy_logits, value_preds

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

        Replaces the kept-cards multi-hot with one set embedding, and the tray
        index columns with per-slot card-table rows plus a tray-set embedding.
        When ``include_turn1_playable`` and/or ``include_playable_kept_cards`` are
        active, each trailing 180-dim multi-hot is embedded as one additional set
        vector (in that order).
        Used by both ``forward`` and ``policy_and_value`` so the embedding is
        computed once regardless of how many heads are read."""
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
        feeder_end_abs = enc.off_turn1_playable
        feeder_goals_rest = features[..., enc.off_feeder : feeder_end_abs]

        # Kept set -> one N-dim embedding (summary derived from the multi-hot).
        kept_summary = hand_model.set_summary_from_multihot(
            kept_multihot, self.card_summary_matrix[1:]
        )
        kept_emb = hand_model.embed_card_set(
            self.hand_encoder, kept_multihot, kept_summary
        )

        # Tray -> 3 M-dim card-table rows + one N-dim set embedding.
        tray_slot_emb = card_table[tray_idx].reshape(*tray_idx.shape[:-1], -1)
        tray_multihot = hand_model.multihot_from_indices(
            tray_idx, encode.HAND_MULTIHOT_DIM
        )
        tray_summary = hand_model.set_summary_from_indices(
            tray_idx, self.card_summary_matrix
        )
        tray_set_emb = hand_model.embed_card_set(
            self.hand_encoder, tray_multihot, tray_summary
        )

        # Embed any appended 180-dim card-set multi-hots (turn1_playable, then
        # playable_kept_cards), each as one extra N-dim set embedding.
        appended: list[torch.Tensor] = []
        for stripe_off in self._appended_multihot_offsets():
            mh = features[..., stripe_off : stripe_off + enc.kept_cards_dim]
            summary = hand_model.set_summary_from_multihot(
                mh, self.card_summary_matrix[1:]
            )
            appended.append(hand_model.embed_card_set(self.hand_encoder, mh, summary))

        return torch.cat(
            [kept_emb, passthrough, tray_set_emb, tray_slot_emb, feeder_goals_rest]
            + appended,
            dim=-1,
        )

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
