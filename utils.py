"""Small shared helpers: resident-memory tracing and JSON coercion."""

try:
    import psutil

    def mem(tag):
        """Print resident set size, so a crash mid-run is visible in the log."""
        print(f"[MEM] {tag}: {psutil.Process().memory_info().rss / 1e9:.2f} GB",
              flush=True)
except ImportError:                                  # psutil is optional
    def mem(tag):
        pass


def jsonable(d):
    """Convert numpy scalars to plain Python so a row dict can be json.dump'ed."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
