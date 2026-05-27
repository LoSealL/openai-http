"""
Bearer token authentication.

Provides FastAPI dependency for `Authorization: Bearer <token>` validation.
Supports open mode (auth disabled) and multi-key validation.
"""

from typing import Callable, Optional
from fastapi import Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from openai_http.errors import AuthenticationError


_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> Optional[str]:
    config = request.app.state.config
    if not config.auth.enabled:
        return None

    if credentials is None:
        raise AuthenticationError(
            message="You must provide an API key.",
            code="missing_api_key",
        )

    token = credentials.credentials
    if not config.auth.api_keys or token not in config.auth.api_keys:
        raise AuthenticationError(
            message="Incorrect API key provided.",
            code="invalid_api_key",
        )

    return token


def create_verify_api_key(config) -> Callable:
    async def _verify(
        credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
    ) -> Optional[str]:
        if not config.auth.enabled:
            return None

        if credentials is None:
            raise AuthenticationError(
                message="You must provide an API key.",
                code="missing_api_key",
            )

        token = credentials.credentials
        if not config.auth.api_keys or token not in config.auth.api_keys:
            raise AuthenticationError(
                message="Incorrect API key provided.",
                code="invalid_api_key",
            )

        return token

    return _verify
