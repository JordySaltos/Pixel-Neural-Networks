# Virtual env instructions

## Windows (Python 3.14)

Create a virtual environment inside your project directory:
```
python -m venv .venv
```

Activate it:
```
.venv\Scripts\activate
```
Your prompt will change to `(.venv)` when the environment is active.

Install project dependencies. Choose the file that matches your setup:
```
# GPU (CUDA 12)
pip install -r requirements_gpu.txt

# CPU only
pip install -r requirements_cpu.txt
```

Deactivate when done:
```
deactivate
```

Recreate the environment from scratch:
```
rmdir /s /q .venv
python -m venv .venv
```

## macOS / Linux

Create and activate a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```
# GPU (CUDA 12)
pip install -r requirements_gpu.txt

# CPU only
pip install -r requirements_cpu.txt
```

Deactivate when done:
```
deactivate
```

Recreate the environment from scratch:
```
rm -rf .venv
python3 -m venv .venv
```

## pre-commit setup

After activating your environment, install the git hooks:
```
pip install pre-commit
pre-commit install
```
Black and flake8 will now run automatically on every `git commit`.
