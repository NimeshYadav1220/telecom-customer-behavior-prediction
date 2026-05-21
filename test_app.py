import streamlit as st

st.set_page_config(page_title="Test App", page_icon="✅")
st.title("✅ Streamlit Test App")
st.write("If you can see this, Streamlit is working correctly!")

# Simple input
name = st.text_input("Enter your name", "")
if name:
    st.success(f"Hello, {name}!")

# Show Streamlit info
st.write("Streamlit version:", st.__version__)
