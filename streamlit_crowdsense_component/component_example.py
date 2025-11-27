import streamlit as st
from streamlit_crowdsense_component import crowdsense_component

st.title('CrowdSense Component Example')

value = crowdsense_component(default={"running": False, "densityThreshold": 65, "sensitivity": 0.5}, key='cs1')
st.write('Component value (from frontend):', value)

if value and isinstance(value, dict):
    if value.get('running'):
        st.info('Component requested to start analysis')
    else:
        st.write('Idle')
