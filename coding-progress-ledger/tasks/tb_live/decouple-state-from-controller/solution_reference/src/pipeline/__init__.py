def process(items: list[dict], config: dict) -> list[dict]:
    accept = config.get("accept_status", "ok")
    fields = config.get("fields", None)
    version = config.get("version", "v1")
    sort_key = config.get("sort_key", None)
    limit = config.get("limit", len(items))

    result = []
    for item in items:
        if item.get("status") != accept:
            continue
        row = {k: v for k, v in item.items() if fields is None or k in fields}
        row = {k: (v.strip().lower() if isinstance(v, str) else v) for k, v in row.items()}
        row["_pipeline_version"] = version
        result.append(row)

    if sort_key is not None:
        result.sort(key=lambda r: r[sort_key])

    return result[:limit]
