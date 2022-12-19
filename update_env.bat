poetry env remove python

:: create / match lock file
poetry lock --no-update

:: install into virtual env from lock
poetry install

:: install jupyter
jupyter contrib nbextension install --user
jupyter nbextension enable toc2/main
jupyter nbextension enable varInspector/main
jupyter nbextension enable execute_time/ExecuteTime
echo y | jupyter kernelspec uninstall quantum_ml

poetry run python -m ipykernel install --user --name=quantum_ml
@REM backup
@REM py -3.8 -m ipykernel install --user --name=quantum_ml 

:: create setup for native package install 
poetry2setup > setup.py