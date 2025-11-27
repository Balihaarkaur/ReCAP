Frontend instructions

This frontend is a minimal React app that implements the CrowdSense controls and communicates changes back to Streamlit when embedded as a component.

Dev workflow:
1. cd into the frontend folder
2. npm install
3. npm run start
   - This starts a webpack dev server at http://localhost:3001
   - When the Python wrapper declares the component with url=http://localhost:3001 Streamlit will load the dev frontend

Build for production:
1. cd into the frontend folder
2. npm install
3. npm run build
   - The build artifacts are written to `frontend/build`
   - The Python wrapper will use `components.declare_component(path=build_dir)` in _RELEASE mode

Notes:
- You need Node.js and npm installed to run the dev server or build the frontend.
- The `streamlit-component-lib` runtime is available when this app is embedded in Streamlit, and the example UI calls `window.Streamlit.setComponentValue(...)` when controls change.
