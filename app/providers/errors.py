class LlmConfigurationError(RuntimeError):
    """The selected provider is not configured."""


class LlmRequestError(RuntimeError):
    """The selected provider failed to complete a request."""


class LlmContextLimitError(LlmRequestError):
    """Required input cannot fit into the configured context budget."""


class LlmTruncatedResponseError(LlmRequestError):
    """A structured response did not fit into one complete answer."""

    def __init__(self, provider: str = "DeepSeek") -> None:
        super().__init__(f"{provider} обрезал JSON по лимиту ответа. Полная проверка не получена")
