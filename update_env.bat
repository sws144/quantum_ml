poetry env remove python
poetry install
jupyter contrib nbextension install --user
jupyter nbextension enable toc2/main
jupyter nbextension enable varInspector/main
jupyter nbextension enable execute_time/ExecuteTime
echo y | jupyter kernelspec uninstall quantum_ml
py -3.8 -m ipykernel install --user --name=quantum_ml
poetry2setup > setup.py