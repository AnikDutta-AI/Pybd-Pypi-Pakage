import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

while True:
    project_name = input("Enter project name: ")
    if project_name:
        break

    else:
        logging.error("Project name cannot be empty")


logging.info(f"creating Project by name: {project_name}")

list_of_files = [
        # GitHub / CI-CD
        ".github/workflows/.gitkeep",
        ".github/workflows/tests.yml",
        ".github/workflows/build.yml",
        #-->".github/workflows/publish.yml",

        # Documentation
        "README.md",
        #-->"CHANGELOG.md",
        #-->"CONTRIBUTING.md",
        #-->"SECURITY.md",
        #-->"CODE_OF_CONDUCT.md",

        # Documentation / examples
        "docs/.gitkeep",
        "examples/.gitkeep",
        #-->"examples/basic_usage.py",
        #-->"examples/advanced_usage.py",

        # Source code
        f"src/{project_name}/__init__.py",
        f"src/{project_name}/main.py",
        f"src/{project_name}/config.py",
        #--->f"src/{project_name}/exceptions.py",
        #--->f"src/{project_name}/utils.py",
        #--->f"src/{project_name}/cli.py",

        # Tests
        "tests/__init__.py",
        "tests/unit/__init__.py",
        "tests/unit/test_main.py",
        #-->"tests/unit/test_config.py",
        #-->"tests/unit/test_utils.py",
        #-->"tests/unit/test_cli.py",

        "tests/integration/__init__.py",
        "tests/integration/test_integration.py",

        # Configuration / dependencies
        "requirements.txt",
        "requirements_dev.txt",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "tox.ini",

        # Project / Git
        ".gitignore",
        ".gitattributes",
        "LICENSE",

        # Optional developer configuration
        ".editorconfig",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir,filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Created directory: {filedir}")
    if(not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,'w') as f:
            pass
            logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")

