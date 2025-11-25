"""Simple test to verify Streamlit works."""
import streamlit as st

st.set_page_config(page_title="Test", page_icon="🔧")
st.title("🔧 Test App")
st.write("If you see this, Streamlit is working!")
st.write(f"Python path: {st.__file__}")
