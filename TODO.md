# Issues with the detailed game log from `wingspan play --log`

1. In the "CHOOSING BIRDS" section, we're still printing "foods:[none]
   bonus:(none)" when we have the initial food/bonus selection split out.
   `decisions.SetupChoice.display_label()` still renders those segments
   unconditionally, so anything that calls it directly for a split-setup
   choice shows the misleading text. `wingspan aid` already works around
   this — `aid/advisor.py`'s `_compact_keep_label` swaps in a compact
   `keep:[...]` label whenever the bonus/food axes are deferred — but
   `wingspan play --log` doesn't: `players/factory.py` still calls
   `choice.display_label()` directly (both in `_log_distribution`'s ranked
   option lines and in the `chose:` line), so the split regime prints the
   stale "keeps nothing" text there. Port the same compact-label logic (or
   share it) into `players/factory.py`.
