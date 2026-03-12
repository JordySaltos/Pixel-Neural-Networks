Create a `docs` folder and go into it (`cd docs`) before calling `sphinx-quickstart`.

Put `"sphinx.ext.napoleon"` in `extensions` inside `config.py`. Also add these lines at the beginning of `config.py`:
````
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
````

Add the modules you want to document inside `index.rst`, don't forget indentation. You can also comment out the `Indices and tables` section.

Run in the console `sphinx-apidoc -f -o source ..`.

Document your code with numpy or google docstrings.

Create your documentation in the console with `make html` or `make latexpdf`. If the ouput of `make html` doesn't seem changed after updating docstrings, use `make clean` before.