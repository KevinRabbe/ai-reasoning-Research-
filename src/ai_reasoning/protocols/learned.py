from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LearnedProtocolSpec:
    """POC-0 discrete machine-native channel contract.

    The trainable encoder is intentionally deferred until the shared neural
    controller is added. The channel itself is frozen now so later results
    cannot move the goalposts.
    """

    vocabulary_size: int = 256
    bits_per_symbol: int = 8
    max_symbols: int = 64
    stop_symbol: int = 0
