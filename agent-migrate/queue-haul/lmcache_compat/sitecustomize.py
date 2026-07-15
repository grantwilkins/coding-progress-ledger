from connector_patch import patch_adapter, patch_lmcache, patch_on_import

patch_lmcache()
patch_on_import("lmcache.integration.vllm.vllm_v1_adapter", patch_adapter)
