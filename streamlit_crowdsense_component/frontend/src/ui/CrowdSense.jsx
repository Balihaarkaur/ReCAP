import React, { useEffect, useState } from 'react';

// The Streamlit component runtime is available when this app is embedded by Streamlit.
// We'll attempt to access the runtime via the global `Streamlit` object.
// See: https://docs.streamlit.io/library/components

const CrowdSense = () => {
  const [state, setState] = useState({ running: false, densityThreshold: 65, sensitivity: 0.5 });

  useEffect(() => {
    // When embedded, Streamlit will send a `render` message containing `args`.
    // If the runtime is present, register handlers.
    if (window.Streamlit) {
      window.Streamlit.events.addEventListener(window.Streamlit.events.RENDER_EVENT, (event) => {
        const args = event.detail.args || {};
        if (args.default) {
          setState((s) => ({ ...s, ...args.default }));
        }
      });

      // notify Streamlit of the current frame height
      window.Streamlit.setFrameHeight(document.body.scrollHeight);
    }
  }, []);

  function pushUpdate(newState) {
    setState(newState);
    // When running as a Streamlit Component, set the component value so Python side receives updates
    if (window.Streamlit && window.Streamlit.setComponentValue) {
      window.Streamlit.setComponentValue(newState);
    }
  }

  return (
    <div style={{ fontFamily: 'Inter, Arial, sans-serif', padding: 16, width: 480 }}>
      <h2 style={{ marginTop: 0 }}>CrowdSense</h2>

      <div style={{ marginBottom: 8 }}>
        <button
          style={{ background: state.running ? '#f66' : '#6f6', padding: '8px 12px', border: 'none', borderRadius: 6 }}
          onClick={() => pushUpdate({ ...state, running: !state.running })}
        >
          {state.running ? 'Stop' : 'Start'}
        </button>
      </div>

      <label>Density Threshold: {state.densityThreshold}%</label>
      <input
        type="range"
        min="1"
        max="100"
        value={state.densityThreshold}
        onChange={(e) => pushUpdate({ ...state, densityThreshold: Number(e.target.value) })}
      />

      <div style={{ marginTop: 12 }}>
        <label>Sensitivity: {state.sensitivity.toFixed(2)}</label>
        <input
          type="range"
          min="0.1"
          max="1.0"
          step="0.05"
          value={state.sensitivity}
          onChange={(e) => pushUpdate({ ...state, sensitivity: Number(e.target.value) })}
        />
      </div>

      <div style={{ marginTop: 12 }}>
        <small style={{ color: '#666' }}>This is a live component. When embedded in Streamlit, changes are sent to the Python side.</small>
      </div>
    </div>
  );
};

export default CrowdSense;
