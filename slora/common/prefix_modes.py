"""Canonical experiment names and backwards-compatible mode aliases."""

FULL_REPREFILL = "full-reprefill"
FULL_RECONSTRUCT = "minikv-full_reconstruct"
LAYER_SERIAL = "minikv-layer_serial"
LAYER_RETAINED = "minikv-layer_retained"
LAYER_PREFETCH = "layer_prefetch"

PREFIX_MODES = (FULL_REPREFILL, FULL_RECONSTRUCT, LAYER_SERIAL, LAYER_PREFETCH, LAYER_RETAINED)
FULL_REPREFILL_MODES = (FULL_REPREFILL,)
FULL_PREFIX_CACHE_MODES = (FULL_REPREFILL, FULL_RECONSTRUCT)

_ALIASES = {mode: mode for mode in PREFIX_MODES}
_ALIASES.update(full=FULL_REPREFILL, full_reprefill=FULL_REPREFILL,
                full_reconstruct=FULL_RECONSTRUCT, layer_serial=LAYER_SERIAL,
                layer_retained=LAYER_RETAINED)


def normalize_prefix_mode(mode):
    try:
        return _ALIASES[mode]
    except KeyError:
        raise ValueError("unknown prefix mode: %s" % mode) from None
