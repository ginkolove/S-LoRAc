"""Canonical experiment names and backwards-compatible mode aliases."""

FULL_REPREFILL = "full-reprefill"
ACTIVE_REPREFILL = "active-reprefill"
CPU_READ = "cpu-read"
IMPRESS = "impress"
CPU_FIXED = "cpu-read-fixed2"
CPU_FIXED_R64 = "cpu-read-fixed1-r64"
CPU_PIN_COUNTS = {CPU_FIXED: 2, CPU_FIXED_R64: 1}
CPU_MODES = (CPU_READ, *CPU_PIN_COUNTS, IMPRESS)
FULL_RECONSTRUCT = "minikv-full_reconstruct"
LAYER_SERIAL = "minikv-layer_serial"
LAYER_RETAINED = "minikv-layer_retained"
LAYER_PREFETCH = "layer_prefetch"
LAYER_ADAPTIVE = "minikv-adaptive"

PREFILL_BUILD_MODES = (FULL_REPREFILL, ACTIVE_REPREFILL)
# Full-size GPU entries produced without a static MiniKV region.
FULL_REPREFILL_MODES = PREFILL_BUILD_MODES + CPU_MODES
FULL_PREFIX_CACHE_MODES = FULL_REPREFILL_MODES + (FULL_RECONSTRUCT,)
PREFIX_MODES = FULL_PREFIX_CACHE_MODES + (LAYER_SERIAL, LAYER_PREFETCH, LAYER_RETAINED, LAYER_ADAPTIVE)
SOURCE_COMPARISON_MODES = (ACTIVE_REPREFILL, CPU_READ, CPU_FIXED, FULL_RECONSTRUCT, LAYER_SERIAL)

_ALIASES = {mode: mode for mode in PREFIX_MODES}
_ALIASES.update(full=FULL_REPREFILL, full_reprefill=FULL_REPREFILL,
                full_reconstruct=FULL_RECONSTRUCT, layer_serial=LAYER_SERIAL,
                layer_retained=LAYER_RETAINED)


def normalize_prefix_mode(mode):
    try:
        return _ALIASES[mode]
    except KeyError:
        raise ValueError("unknown prefix mode: %s" % mode) from None


def add_impress_read_argument(parser):
    parser.add_argument("--impress-read-percent", type=int, choices=range(101), metavar="0..100",
                        help="Percentage of prefix positions that Impress copies from CPU (default 50).")
