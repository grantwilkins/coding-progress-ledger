from connector_patch import patch_adapter, patch_lmcache
import os

if os.environ.get("QH_LMCACHE_MODE", "legacy") == "legacy":
    patch_lmcache()
    patch_adapter()
