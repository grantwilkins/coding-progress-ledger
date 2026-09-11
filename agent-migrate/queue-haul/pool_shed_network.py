"""Fixed migration reservations; idle host/history shares are never borrowed."""

import numpy as np


def _fixed_settings(fleet):
    enabled = fleet.metadata.get("fixed_host_shares", False)
    if enabled is False:
        return None
    if enabled is not True:
        raise ValueError("fixed_host_shares must be a boolean")
    source, destination = fleet.gpus, fleet.metadata.get("destination_gpus", fleet.gpus)
    count = np.asarray(fleet.count, float)
    gbps = fleet.metadata.get("host_migration_gbps", 80.)
    if (any(not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0
            for n in (source, destination)) or fleet.gpus_per_node != 8
            or fleet.metadata.get("resident_affinity") is not True
            or count.ndim != 1 or not np.isfinite(count).all() or np.any(count < 0)
            or count.sum() > 8 * source + 1e-8
            or isinstance(gbps, (bool, np.bool_)) or np.ndim(gbps) != 0 or not np.isfinite(gbps) or gbps <= 0):
        raise ValueError("fixed host shares require eight GPUs/host, at most eight source histories/GPU, resident affinity and positive bandwidth")
    return source, destination, float(gbps) * 1e9 / 8


def transport_workers(fleet):
    """Legacy node pools, or pooled per-GPU worker equivalents (unmeasured scaling)."""
    fixed = _fixed_settings(fleet)
    if fixed is None:
        from pool_shed_execution import network_nodes
        return network_nodes(fleet)
    source, destination, _ = fixed
    return np.array([min(source, destination), min(source, destination), source])


def phase_rate_cap(fleet, counts, per_history_bytes):
    """Per-replica B/s cap for proportional sending; repeated counts are distinct histories."""
    fixed = _fixed_settings(fleet)
    if fixed is None:
        return np.inf
    counts, volume = np.asarray(counts, float), np.asarray(per_history_bytes, float)
    if (counts.shape != np.shape(fleet.count) or volume.shape != counts.shape
            or not np.isfinite(np.r_[counts, volume]).all() or np.any(counts < 0)
            or np.any(counts != np.floor(counts)) or np.any(volume < 0)):
        raise ValueError("phase bytes and integer history counts must be aligned, finite and nonnegative")
    host_rate = fixed[2]
    largest = np.max(volume[counts > 0], initial=0.)
    # Every history owns 1/64 of source egress; each destination GPU owns 1/8 of ingress.
    return min(host_rate / 8, host_rate / 64 * float(counts @ volume) / largest) if largest else host_rate / 8


def history_bytes(fleet, contexts, action, calibration, origin_context=None, reset=None):
    """Match initial/catch-up wire primitives; a reset mask selects the catch-up phase."""
    from pool_shed_execution import kv_transfer_bytes

    context = np.asarray(fleet.context if contexts is None else contexts, float)
    if (action not in (0, 1) or context.shape != np.shape(fleet.count)
            or not np.isfinite(context).all() or np.any(context < 0)):
        raise ValueError("network action and source contexts must be valid")
    if reset is None:
        if origin_context is not None:
            raise ValueError("catch-up origins require an explicit reset mask")
        volume = np.where(context == fleet.context, fleet.kv if action else fleet.log,
                          kv_transfer_bytes(fleet, context, calibration) if action else 2 * context)
    else:
        origin = np.asarray(fleet.context if origin_context is None else origin_context, float)
        reset = np.asarray(reset)
        if (origin.shape != context.shape or not np.isfinite(origin).all() or np.any(origin < 0)
                or reset.shape != context.shape or reset.dtype != bool):
            raise ValueError("catch-up origin and boolean reset mask must match source contexts")
        if action:
            old = np.where(reset, 0, kv_transfer_bytes(fleet, origin, calibration))
            volume = np.maximum(kv_transfer_bytes(fleet, context, calibration) - old, 0)
        else:
            volume = np.where((context > 0) & (reset | (context != origin)), 2 * context, 0)
    if volume.shape != context.shape or not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("per-history wire bytes must be aligned, finite and nonnegative")
    return volume
