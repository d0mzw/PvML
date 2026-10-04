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
        # raise, not assert: python -O strips asserts, and these are a guarantee
        # the sampler relies on rather than a development-time check.
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if not 0 <= self.top_p <= 1.0:
            raise ValueError(f"top_p must be a probability in [0, 1], got {self.top_p}")
        # They are alternatives, not a pair. Allowing both would silently apply
        # whichever the dispatcher happens to check first.
        if self.top_k and self.top_p:
            raise ValueError(
                f"set at most one of top_k and top_p, got top_k={self.top_k} top_p={self.top_p}"
            )

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


if __name__ == "__main__":
    print(f"defaults: {SamplingArgs().describe()}\n")

    # Every condition in __post_init__, each one checked.
    cases = [
        ("max_new_tokens <= 0", dict(max_new_tokens=0)),
        ("temperature < 0", dict(temperature=-0.5)),
        ("top_k < 0", dict(top_k=-1)),
        ("top_p outside [0, 1]", dict(top_p=1.5)),
        ("top_k and top_p together", dict(top_k=40, top_p=0.95)),
    ]
    for label, kwargs in cases:
        try:
            SamplingArgs(**kwargs)
            print(f"  FAIL  {label:<26} accepted {kwargs}")
        except ValueError as e:
            print(f"  ok    {label:<26} {e}")

    # The valid edges must still be accepted.
    for label, kwargs in [
        ("temperature 0 (greedy)", dict(temperature=0.0)),
        ("top_p 1.0", dict(top_p=1.0)),
        ("top_k only", dict(top_k=40)),
    ]:
        SamplingArgs(**kwargs)
        print(f"  ok    {label:<26} accepted")
