from __future__ import annotations

from chromhandler.readers.agilent import AgilentReader
from chromhandler.readers.asm import ASMReader
from chromhandler.readers.knauer_txt import KnauerTXTReader
from chromhandler.readers.shimadzu import ShimadzuReader

# Ordered probe list for Handler.read() auto-detection.
# Agilent and ASM are checked first (directory scan, no file I/O).
# Knauer precedes Shimadzu (5-line sniff vs. full-file scan); their
# signatures are mutually exclusive so order does not affect correctness.
READERS: list[type] = [
    AgilentReader,
    ASMReader,
    KnauerTXTReader,
    ShimadzuReader,
]

__all__ = ["READERS", "ASMReader", "AgilentReader", "KnauerTXTReader", "ShimadzuReader"]
