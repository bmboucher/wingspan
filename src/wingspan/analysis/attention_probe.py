"""A manually re-implemented, perturbable stand-in for board self-attention.

``model.core.PolicyValueNet`` reads its board-attention module(s)
(``board_attn`` when shared, or the ``board_attn_me`` / ``board_attn_opp``
pair) purely through the ``nn.MultiheadAttention`` call contract —
``module(query, key, value, key_padding_mask=..., need_weights=False)``
returning ``(output, weights)``, plus the module's ``embed_dim`` / ``num_heads``
attributes (``model.core._apply_board_attention``). :class:`ProbeAttention`
duck-types that exact contract so it can be installed in place of the real
module without touching ``model/core.py`` — the padding, masking, and
empty-slot residual all stay owned by ``_apply_board_attention``; this module
only needs to reproduce (or perturb) honest multi-head self-attention.

:func:`install` / :func:`uninstall` swap the wrapper(s) in and out of a net;
:func:`set_mode` flips every installed wrapper to the same
:class:`models.AblationMode` in one call. :class:`AttentionStatsCollector`
accumulates FULL-mode entropy / self-weight / contribution statistics across
every board a probe pass visits, for :func:`representation.measure` to roll
into one :class:`models.AttentionBlockStats`.
"""

from __future__ import annotations

import math
import typing

import torch
import torch.nn.functional as F
from torch import nn

from wingspan import encode, model
from wingspan.analysis import models

# Sentinel ``knockout_head`` meaning "no head knocked out" — the default for
# every mode except HEAD_KNOCKOUT.
_KNOCKOUT_DISABLED = -1

# Softmax masking value: a key masked at this score receives ~0 probability.
_NEGATIVE_INFINITY = float("-inf")

# Entropy-collection guard: a row with fewer than this many filled slots has
# no meaningful "spread" to measure (a single unmasked key is entropy 0 by
# construction, not evidence of a peaked head).
_MIN_FILLED_FOR_ENTROPY = 2

# Numerical floors so log(0) and division-by-zero never appear in the stats.
_ENTROPY_LOG_EPS = 1e-12
_MIN_TOKEN_NORM = 1e-6

# Percentiles the report's "spread" readouts are computed at.
_LOW_QUANTILE = 0.1
_HIGH_QUANTILE = 0.9


class ProbeAttention(nn.Module):
    """Drop-in replacement for one board-attention ``nn.MultiheadAttention``.

    Reproduces the module's forward pass by hand (in-projection split into
    per-head q/k/v, scaled dot-product scores, key-padding mask, softmax,
    out-projection) so ``self.mode`` can substitute a perturbed computation:
    ``ZERO`` returns an all-zero output, ``UNIFORM`` replaces the learned
    attention weights with a uniform average over the unmasked keys,
    ``HEAD_KNOCKOUT`` zeroes exactly ``self.knockout_head``'s contribution
    before the heads are merged, and ``FULL`` (the default) matches the
    original module exactly. ``true_width`` is the token width before the
    caller's zero-padding (``architecture.board_attention_embed_dim``) — used
    only to slice the "real" leading columns for :attr:`collector`'s
    contribution-ratio / cosine statistics, never for the attention math
    itself, which always operates at the full (possibly padded) ``embed_dim``.
    """

    def __init__(
        self,
        attention: nn.MultiheadAttention,
        true_width: int,
        collector: "AttentionStatsCollector | None" = None,
    ) -> None:
        super().__init__()
        self.wrapped = attention
        # nn.MultiheadAttention leaves embed_dim/num_heads unannotated in the
        # torch stubs (Unknown under strict pyright); core.py's
        # _apply_board_attention casts the same two attributes for the same
        # reason, so this call site reads config.arch off the live weights.
        self.embed_dim = typing.cast(int, attention.embed_dim)
        self.num_heads = typing.cast(int, attention.num_heads)
        self.true_width = true_width
        self.mode = models.AblationMode.FULL
        self.knockout_head = _KNOCKOUT_DISABLED
        self.collector = collector

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, None]:
        """Self-attend over ``query`` (``key`` / ``value`` are the same tensor
        for board attention; accepted only to match the call contract).
        Returns ``(output, None)`` — never returns attention weights, matching
        every call site's ``need_weights=False``."""
        tokens = query
        if self.mode == models.AblationMode.ZERO:
            return torch.zeros_like(tokens), None

        # Manual in-projection + per-head split, mirroring
        # nn.MultiheadAttention's internal math for self-attention (q=k=v).
        batch_size, seq_len, embed_dim = tokens.shape
        head_dim = embed_dim // self.num_heads
        projected = F.linear(
            tokens, self.wrapped.in_proj_weight, self.wrapped.in_proj_bias
        )
        query_heads, key_heads, value_heads = projected.chunk(3, dim=-1)
        query_heads = _split_heads(query_heads, self.num_heads, head_dim)
        key_heads = _split_heads(key_heads, self.num_heads, head_dim)
        value_heads = _split_heads(value_heads, self.num_heads, head_dim)

        # Scaled dot-product scores, masked so a padded key never receives
        # weight; UNIFORM then discards the softmax in favor of an average
        # over whatever keys survived the mask.
        scores = (query_heads @ key_heads.transpose(-1, -2)) / math.sqrt(head_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], _NEGATIVE_INFINITY
            )
        attention_weights = scores.softmax(dim=-1)
        if self.mode == models.AblationMode.UNIFORM:
            attention_weights = _uniform_over_filled(
                attention_weights, key_padding_mask
            )

        # Per-head weighted sum, with HEAD_KNOCKOUT zeroing one head's output
        # before the heads are merged back through the shared out-projection.
        head_outputs = attention_weights @ value_heads
        if self.mode == models.AblationMode.HEAD_KNOCKOUT:
            head_outputs = head_outputs.clone()
            head_outputs[:, self.knockout_head] = 0.0
        merged = head_outputs.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = F.linear(merged, self.wrapped.out_proj.weight, self.wrapped.out_proj.bias)

        if (
            self.collector is not None
            and self.mode == models.AblationMode.FULL
            and key_padding_mask is not None
        ):
            self.collector.record(
                attention_weights.detach(),
                out.detach(),
                tokens.detach(),
                key_padding_mask,
            )
        return out, None


class AttentionStatsCollector:
    """Accumulates FULL-mode board-attention statistics across every board a
    probe pass visits (the POV board, then each opponent's — one or two
    :class:`ProbeAttention` wrappers may report into the same collector), for
    roll-up into one :class:`models.AttentionBlockStats` via :meth:`summarize`.

    Not an ``nn.Module`` — a plain accumulator a wrapper's ``collector``
    attribute points at.
    """

    def __init__(self, num_heads: int, true_width: int) -> None:
        self._true_width = true_width
        self._entropy_by_head: list[list[torch.Tensor]] = [[] for _ in range(num_heads)]
        self._self_weight_by_head: list[list[torch.Tensor]] = [
            [] for _ in range(num_heads)
        ]
        self._contribution_ratios: list[torch.Tensor] = []
        self._cosines: list[torch.Tensor] = []

    def record(
        self,
        attention_weights: torch.Tensor,
        out: torch.Tensor,
        tokens: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> None:
        """Fold one board's FULL-mode attention weights, output, and input
        tokens into the running statistics.

        ``attention_weights`` is ``(B, H, T, T)``; ``out`` / ``tokens`` are
        ``(B, T, E)``; ``key_padding_mask`` is ``(B, T)`` (True = masked/empty)."""
        self._record_entropy_and_self_weight(attention_weights, key_padding_mask)
        self._record_contribution(out, tokens, key_padding_mask)

    def summarize(self) -> models.AttentionBlockStats:
        """Roll the accumulated statistics into one
        :class:`models.AttentionBlockStats`. Every recorded quantity must have
        at least one sample — callers only construct a collector when at
        least one FULL-mode pass over a non-empty probe set has run."""
        ratio = torch.cat(self._contribution_ratios)
        cosine = torch.cat(self._cosines)
        heads = [
            self._summarize_head(head_index)
            for head_index in range(len(self._entropy_by_head))
        ]
        return models.AttentionBlockStats(
            contribution_ratio_median=float(ratio.median()),
            contribution_ratio_p10=float(ratio.quantile(_LOW_QUANTILE)),
            contribution_ratio_p90=float(ratio.quantile(_HIGH_QUANTILE)),
            cosine_median=float(cosine.median()),
            cosine_p10=float(cosine.quantile(_LOW_QUANTILE)),
            cosine_p90=float(cosine.quantile(_HIGH_QUANTILE)),
            heads=heads,
        )

    ###### PRIVATE #######

    def _record_entropy_and_self_weight(
        self, attention_weights: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> None:
        """Per-head normalized entropy and self-weight, restricted to rows
        with at least ``_MIN_FILLED_FOR_ENTROPY`` filled keys."""
        filled = ~key_padding_mask
        n_filled = filled.sum(1)
        rows = filled & (n_filled >= _MIN_FILLED_FOR_ENTROPY)[:, None]
        if not bool(rows.any()):
            return
        entropy = -(
            attention_weights.clamp_min(_ENTROPY_LOG_EPS).log() * attention_weights
        ).sum(-1)
        normalizer = torch.log(n_filled.float().clamp(min=_MIN_FILLED_FOR_ENTROPY))
        normalized_entropy = entropy / normalizer[:, None, None]
        self_weight = attention_weights.diagonal(dim1=-2, dim2=-1)
        for head_index in range(attention_weights.shape[1]):
            self._entropy_by_head[head_index].append(
                normalized_entropy[:, head_index][rows]
            )
            self._self_weight_by_head[head_index].append(
                self_weight[:, head_index][rows]
            )

    def _record_contribution(
        self, out: torch.Tensor, tokens: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> None:
        """``||out|| / ||token||`` and ``cos(out, token)`` on the true (unpadded)
        leading columns, restricted to filled slots with a non-degenerate token."""
        filled = ~key_padding_mask
        out_true = out[..., : self._true_width]
        tokens_true = tokens[..., : self._true_width]
        out_norm = _row_norm(out_true)
        token_norm = _row_norm(tokens_true)
        selected = filled & (token_norm > _MIN_TOKEN_NORM)
        self._contribution_ratios.append(
            (out_norm / token_norm.clamp(min=_MIN_TOKEN_NORM))[selected]
        )
        self._cosines.append(
            F.cosine_similarity(out_true, tokens_true, dim=-1)[selected]
        )

    def _summarize_head(self, head_index: int) -> models.AttentionHeadStats:
        entropy = torch.cat(self._entropy_by_head[head_index])
        self_weight = torch.cat(self._self_weight_by_head[head_index])
        return models.AttentionHeadStats(
            head=head_index,
            entropy_median=float(entropy.median()),
            entropy_p10=float(entropy.quantile(_LOW_QUANTILE)),
            self_weight_median=float(self_weight.median()),
        )


def install(net: model.PolicyValueNet) -> list[ProbeAttention]:
    """Wrap ``net``'s board-attention module(s) in :class:`ProbeAttention` and
    assign the wrapper(s) back onto ``net`` in place of the originals.

    Returns ``[]`` when ``net.arch.use_board_attention`` is off — nothing to
    wrap. Under a shared architecture, one wrapper replaces ``board_attn``;
    otherwise two replace ``board_attn_me`` / ``board_attn_opp`` (POV wrapper
    first, matching ``PolicyValueNet._board_attention_modules``'s
    ``(pov, opponent)`` order). Pair with :func:`uninstall` to restore the
    originals afterward."""
    if not net.arch.use_board_attention:
        return []
    true_width = _true_width_for(net)
    if net.arch.board_attention_shared_active:
        wrapper = ProbeAttention(net.board_attn, true_width)
        net.board_attn = typing.cast(nn.MultiheadAttention, wrapper)
        return [wrapper]
    pov_wrapper = ProbeAttention(net.board_attn_me, true_width)
    opponent_wrapper = ProbeAttention(net.board_attn_opp, true_width)
    net.board_attn_me = typing.cast(nn.MultiheadAttention, pov_wrapper)
    net.board_attn_opp = typing.cast(nn.MultiheadAttention, opponent_wrapper)
    return [pov_wrapper, opponent_wrapper]


def uninstall(net: model.PolicyValueNet, wrappers: list[ProbeAttention]) -> None:
    """Restore the original board-attention module(s) :func:`install`
    replaced. A no-op when ``wrappers`` is empty (an attention-off net)."""
    if not wrappers:
        return
    if len(wrappers) == 1:
        net.board_attn = wrappers[0].wrapped
        return
    net.board_attn_me = wrappers[0].wrapped
    net.board_attn_opp = wrappers[1].wrapped


def set_mode(
    wrappers: list[ProbeAttention],
    mode: models.AblationMode,
    head: int = _KNOCKOUT_DISABLED,
) -> None:
    """Set every installed wrapper to ``mode`` (and, for
    :attr:`models.AblationMode.HEAD_KNOCKOUT`, ``head``) in one call."""
    for wrapper in wrappers:
        wrapper.mode = mode
        wrapper.knockout_head = head


###### PRIVATE #######


def _true_width_for(net: model.PolicyValueNet) -> int:
    """The board-attention token width before the module's own zero-padding
    (``card_embed_dim`` scalars ⊕ the mutable per-slot stripe, plus the
    constant position block when active) — see
    ``model.core._build_board_attention``'s ``token_dim`` derivation."""
    width = net.arch.card_embed_dim + encode.SLOT_SCALAR_DIM
    if net.arch.board_attention_positions_active:
        width += encode.BOARD_POSITION_DIM
    return width


# Line length pinned by hand — black's own reformatting would relocate the
# trailing ignore comment away from the line pyright anchors its diagnostic
# to; `# fmt: off` keeps black from ever re-exploding this.
# fmt: off
def _row_norm(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor.norm(dim=-1)``, pinned to Tensor (the stub's overload
    resolution for ``norm`` returns ``Unknown | Tensor``)."""
    return tensor.norm(dim=-1)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
# fmt: on


def _split_heads(tensor: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """``(B, T, E) -> (B, H, T, d)``: split the last dim into per-head chunks
    and move the head axis next to batch, matching
    ``nn.MultiheadAttention``'s internal layout."""
    batch_size, seq_len, _ = tensor.shape
    return tensor.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def _uniform_over_filled(
    attention_weights: torch.Tensor, key_padding_mask: torch.Tensor | None
) -> torch.Tensor:
    """Replace ``attention_weights`` with a uniform distribution over each
    row's unmasked (filled) keys — the UNIFORM ablation. Falls back to a
    uniform distribution over every key when no mask is given."""
    if key_padding_mask is None:
        return torch.full_like(attention_weights, 1.0 / attention_weights.shape[-1])
    keep = (~key_padding_mask).float()[:, None, None, :].expand_as(attention_weights)
    return keep / keep.sum(-1, keepdim=True).clamp(min=1.0)
