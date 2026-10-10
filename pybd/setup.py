import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.1"

REPO_NAME = "Pybd-Pypi-Pakage"
AUTHOR_USER_NAME = "AnikDutta-AI"
AUTHOR_EMAIL = "anikdatta@gmail.com"

SRC_REPO = "pybd"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,

    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,

    description="A Python package for ...",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url=f"https://github.com/{{AUTHOR_USER_NAME}}/{{REPO_NAME}}",

    project_urls={
        "Bug Tracker": f"https://github.com/{{AUTHOR_USER_NAME}}/{{REPO_NAME}}/issues",
        "Source Code": f"https://github.com/{{AUTHOR_USER_NAME}}/{{REPO_NAME}}",
    },

    package_dir={"": "src"},

    packages=setuptools.find_packages(
        where="src"
    ),

    python_requires=">=3.9",
)