"""
streamlit_crowdsense_component

Python wrapper for the CrowdSense Streamlit Component.

This wrapper declares the component in two modes:
- development mode: connects to a frontend dev server (http://localhost:3001)
- production mode: serves a pre-built frontend from `frontend/build`

Usage:
from streamlit_crowdsense_component import crowdsense_component
value = crowdsense_component(default={"status":"idle"}, key="cs1")
"""
import os
import streamlit.components.v1 as components

_RELEASE = True

# Be defensive: only declare the Streamlit component if the frontend build exists
# (production) or if we are deliberately using the dev server. Otherwise expose
# a no-op Python fallback so importing `streamlit_crowdsense_component` never
# causes the Streamlit frontend to try and load missing assets and show an error
# in the browser.
_component_func = None
_component_available = False

# Allow explicit opt-in via environment variable to avoid surprising client-side
# component load failures. Set `CROWDSENSE_ENABLE_COMPONENT=1` to enable.
_enable_env = os.environ.get("CROWDSENSE_ENABLE_COMPONENT", "0")
_enabled_by_env = str(_enable_env).lower() in ("1", "true", "yes")

try:
    if not _RELEASE and _enabled_by_env:
        # During development you can run the frontend dev server (npm start)
        try:
            _component_func = components.declare_component(
                "crowdsense",
                url="http://localhost:3001",
            )
            _component_available = True
        except Exception:
            # dev server not reachable; fall back to no-op
            _component_func = None
            _component_available = False
    elif _RELEASE and _enabled_by_env:
        parent_dir = os.path.dirname(__file__)
        build_dir = os.path.join(parent_dir, "frontend", "build")
        if os.path.isdir(build_dir):
            try:
                _component_func = components.declare_component(
                    "crowdsense", path=build_dir
                )
                _component_available = True
            except Exception:
                _component_func = None
                _component_available = False
        else:
            # build artifacts missing — don't attempt to declare the component
            _component_func = None
            _component_available = False
except Exception:
    _component_func = None
    _component_available = False


def _no_op_component(default=None, key=None):
    """A safe fallback used when the frontend component isn't available.

    Returning `None` lets calling code fall back to native Streamlit controls.
    """
    return None


def crowdsense_component(default=None, key=None):
    """Call the underlying component if available; otherwise return None.

    - `default` is an arbitrary JSON-serializable dict passed to the frontend as initial state.
    - returns the (JSON-serializable) value sent by the frontend, or `None` when
      the frontend isn't available.
    """
    func = _component_func if _component_available and _component_func is not None else _no_op_component
    try:
        return func(default=default, key=key)
    except Exception:
        # If the component fails at call time, swallow the error and fall back.
        return None
