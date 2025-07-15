After building as .exe, in dist/main, click into /_internal and move .streamlit and main_ui.py into the same directory as main.exe

aka from this:
/dist
  |-- /main
       |-- /_internal  
	    |-- /.streamlit
	    |-- ...
	    |-- main_ui.py
       |-- main.exe

to this:
/dist
  |-- /main
       |-- /_internal      
       |-- main.exe
       |-- main_ui.py
       |-- /.streamlit

An extra directory named logs may appear as well but users besides devs can ignore that