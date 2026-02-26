def assess_component_health(row):
    components = {col: row[col] for col in row.index if 'component_' in col}
    failed = [c for c, v in components.items() if v < 0.4]
    warning = [c for c, v in components.items() if 0.4 <= v < 0.7]

    if failed:
        status = f"⚠️ Likely failure: {', '.join(failed)}"
    elif warning:
        status = f"🟠 Degrading: {', '.join(warning)}"
    else:
        status = "✅ All components healthy"
    return status