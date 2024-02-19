import sys

def is_venv_activated():
    return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

if is_venv_activated():
    print("Virtual environment is activated.")
else:
    print("Virtual environment is not activated.")
