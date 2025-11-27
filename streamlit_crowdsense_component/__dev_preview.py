import os
import streamlit as st
from streamlit.components.v1 import declare_component

parent = os.path.dirname(__file__)
build_dir = os.path.join(parent, 'frontend', 'build')
_comp = declare_component('crowdsense_preview', path=build_dir)

def crowdsense_preview(default=None, key=None):
    return _comp(default=default, key=key)

# demo
st.title('CrowdSense Component (preview build)')
val = crowdsense_preview(default={'running':False,'densityThreshold':65,'sensitivity':0.5}, key='pv')
st.write('Component value:', val)
