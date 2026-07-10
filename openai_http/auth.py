"""
Copyright (C) 2026 The OPENAI-HTTP Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Bearer token authentication.

Provides FastAPI dependency for ``Authorization: Bearer <token>`` validation.
Supports open mode (auth disabled) and multi-key validation.
"""

from typing import Optional

from fastapi import Request, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .errors import AuthenticationError


_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> Optional[str]:
    """Verify the API key from the request's Authorization header.

    If authentication is disabled in config, returns None immediately.

    Args:
        request: The incoming HTTP request.
        credentials: Bearer token credentials extracted by FastAPI.

    Returns:
        The validated API token string, or None if auth is disabled.

    Raises:
        AuthenticationError: If the token is missing or invalid.
    """
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
