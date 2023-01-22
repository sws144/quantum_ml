# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['quantumtools']

package_data = \
{'': ['*']}

install_requires = \
['matplotlib>=3.1.3,<4.0.0',
 'pandas>=1.0.0,<2.0.0',
 'scikit-learn>=1.0.0,<2.0.0']

setup_kwargs = {
    'name': 'quantumtools',
    'version': '0.1.3',
    'description': '',
    'long_description': '# quantumtools\n\nmachine learning utilities\n\n## User Setup\n1. `pip install quantumtools`\n\n## Development setup\n\n1. **Note** prepend `py -3.8 -m` if there are different pythonv versions\n1. Install poetry\n    1. `pip install poetry` # develop in python 3.8\n1. **Note** in windows and vscode, need to ensure vscode env path is \n    1. `~/.virtualenvs` \n    1. `"python.venvPath": "~/.virtualenvs"` in settings.json\n    1. and `py -3.8 -m poetry config virtualenvs.in-project false`\n    1. `py -3.8 -m poetry config virtualenvs.path "~\\.virtualenvs"`\n1. to recreate env with updated pyproject.toml\n    1. enter python env with `py -3.8 -m poetry shell`\n    1. `update_env.bat`\n    1. `pytest`\n1. to add dependencies\n    1. enter python env `py -3.8 -m poetry shell`\n    1. `poetry add packagename@^#.#.#` , @ is for version increase\n    1. for development dependencies `poetry add --group dev packagename@^#.#.#` \n\n## Dev Reference\n\n1. Generate pyproject.toml interactively and create venv\n1. poetry init  or poetry new packagename\n1. Add dependency\n    1. poetry add requests # Add --group dev for development dependencies\n1. Install dependencies into venv\n    1. poetry install\n1. Activate venv\n    1. poetry shell\n1. Exit venv\n    1. exit\n\n1. **NOTE**, to add for installs from github, use `poetry2setup > setup.py`\n',
    'author': 'Stanley Wang',
    'author_email': 's.wang4331@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8,<4.0',
}


setup(**setup_kwargs)

