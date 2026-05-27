import abc
from typing import AsyncGenerator, Optional, Any


class BackendBase(abc.ABC):

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> dict:
        ...

    @abc.abstractmethod
    async def generate_stream(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        ...
        yield ""

    @abc.abstractmethod
    async def list_models(self) -> list[dict]:
        ...

    @abc.abstractmethod
    async def get_model(self, model_id: str) -> Optional[dict]:
        ...

    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        raise NotImplementedError("Embeddings are not supported by this backend")

    async def generate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("Tool calls are not supported by this backend")

    async def setup(self) -> None:
        pass

    async def teardown(self) -> None:
        pass
