"""Keep vLLM server-info collection bounded to runtime configuration."""

from vllm.entrypoints.serve.instrumentator import server_info


server_info._get_system_env_info_cached = lambda: {}


class ConfigOnly:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        await self.app(scope, receive, send)
