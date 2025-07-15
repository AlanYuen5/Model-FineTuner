from PyInstaller.utils.hooks import copy_metadata

# Copy Streamlit's package metadata so importlib.metadata can find it at runtime
datas = copy_metadata('streamlit') 