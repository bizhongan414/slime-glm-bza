import asyncio


def get_event_loop():
    """Get or create an event loop (following verl pattern)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop