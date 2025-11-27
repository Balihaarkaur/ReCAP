CrowdSense Streamlit Component scaffold

What I created:
- `streamlit_crowdsense_component/__init__.py` - Python wrapper that declares the component (development and production modes)
- `streamlit_crowdsense_component/frontend` - React-based frontend skeleton (webpack + React)
- `streamlit_crowdsense_component/frontend/src/ui/CrowdSense.jsx` - the component UI (sliders + Start/Stop controls)

How to develop locally (recommended):

Prerequisites:
- Node.js (>=16) and npm
- Python (your project venv)

1) Start the frontend dev server (in a separate terminal):

```bash
cd streamlit_crowdsense_component/frontend
npm install
npm run start
```

This will host the frontend at `http://localhost:3001`.

2) Run Streamlit in development mode (the Python wrapper defaults to dev mode):

```bash
# from project root, with your venv activated
python -m streamlit run app.py
```

3) Embed the component in your app (example usage is in `component_example.py`):

```python
from streamlit_crowdsense_component import crowdsense_component

value = crowdsense_component(default={"running": False, "densityThreshold": 65})
st.write("Component value:", value)
```

Build for production:

```bash
cd streamlit_crowdsense_component/frontend
npm install
npm run build
# then set _RELEASE=True in streamlit_crowdsense_component/__init__.py (or edit to use path)
```

Notes and caveats:
- The scaffold uses a minimal webpack + React setup. It is a starting point — you may prefer to use Create React App or Vite for a richer tooling environment.
- The frontend uses `window.Streamlit.setComponentValue(...)` to send control updates back to Python. This will only work when the app is embedded in Streamlit via `declare_component`.
- If you want, I can now run the frontend dev server and then run Streamlit so you have a hot-reload two-way UI. Say "run component" and I'll start both (I will run `npm install` & `npm run start` then start Streamlit).