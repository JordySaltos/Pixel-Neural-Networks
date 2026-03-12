# Virtual env instructions

References: [https://opensource.com/article/19/5/python-3-default-mac](https://opensource.com/article/19/5/python-3-default-mac) and [https://opensource.com/article/19/6/python-virtual-environments-mac](https://opensource.com/article/19/6/python-virtual-environments-mac).

Install `virtualenvwrapper` with `pip install virtualenvwrapper`. Then run in a Terminal
```
# We want to regularly go to our virtual environment directory
$ echo 'export WORKON_HOME=~/.virtualenvs' >> .bash_profile
# If in a given virtual environment, make a virtual environment directory
# If one does not already exist
$ echo 'mkdir -p $WORKON_HOME' >> .bash_profile
# Activate the new virtual environment by calling this script
$ echo '. PYTHON/bin/virtualenvwrapper.sh' >> .bash_profile
```
or
```
# We want to regularly go to our virtual environment directory
$ echo 'export WORKON_HOME=~/.virtualenvs' >> .zprofile
# If in a given virtual environment, make a virtual environment directory
# If one does not already exist
$ echo 'mkdir -p $WORKON_HOME' >> .zprofile
# Activate the new virtual environment by calling this script
$ echo '. PYTHON/bin/virtualenvwrapper.sh' >> .zprofile
```
where `PYTHON` is the location of your python installation, for example '~/.pyenv/versions/$(cat ~/.pyenv/version)' if you manage your Python installation with pyenv.
Then refresh your Terminal or open a new Terminal window.

Create your environments with `mkvirtualenv`, switch between them or list them with `workon`, deactivate them with `deactivate`.

Inside the directory of your project you can run 
```
mkvirtualenv $(basename "$(pwd)")
```
to create a virtual env with the same name as your directory, so when you are inside this directory you can simply activate the corresponding environment with 
```
workon .
```

You can then run `pip install -r requirements.txt` to install your project dependencies and/or `python -m ipykernel install --user --name=projectname` to add your virtual env to your jupyter kernels (you need to install `ipykernel` first).

Delete a virtualenv with `rmvirtualenv`: 
```
rmvirtualenv $(basename "$(pwd)")
```
For example you can recreate a project-specific virtual env with:
```
rmvirtualenv $(basename "$(pwd)")
mkvirtualenv $(basename "$(pwd)")
```
