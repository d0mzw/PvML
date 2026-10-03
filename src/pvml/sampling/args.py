from dataclasses import dataclass


@dataclass
class SamplingArgs:
    """How to pick the next token. Architecture lives in Config, run knobs in
    TrainingArgs; these are separate again on purpose.

    The checks run once here rather than on every token, which is where ARENA
    puts them. A bad combination fails before generation starts instead of
    50 forward passes in.
    """

    max_new_tokens: int = 50
    temperature: float = 1.0  # 0 means greedy
    top_k: int = 0  # 0 disables
    top_p: float = 0.0  # 0 disables
    frequency_penalty: float = 0.0
    seed: int | None = None  # set for reproducible draws

    def __post_init__(self) -> None:
        assert self.max_new_tokens > 0, "max_new_tokens must be positive"
        assert self.temperature >= 0, "temperature must be non-negative"
        assert self.top_k >= 0, "top_k must be non-negative"
        assert 0 <= self.top_p <= 1.0, "top_p must be a probability"
        # They are alternatives, not a pair. Allowing both would silently apply
        # whichever the dispatcher happens to check first.
        assert not (self.top_k and self.top_p), "set at most one of top_k and top_p"

    def describe(self) -> str:
        if self.temperature == 0:
            return f"greedy, {self.max_new_tokens} tokens"
        parts = [f"temp {self.temperature}"]
        if self.top_k:
            parts.append(f"top_k {self.top_k}")
        if self.top_p:
            parts.append(f"top_p {self.top_p}")
        if self.frequency_penalty:
            parts.append(f"freq_penalty {self.frequency_penalty}")
        parts.append(f"{self.max_new_tokens} tokens")
        return ", ".join(parts)
