from setuptools import setup, find_packages

def parseRequirements(filename):
    with open(filename, encoding="utf-8") as f:
        lines = f.readlines()
    reqs = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs

setup(
    name="Popcorn",
    version="1.5.0",
    maintainer='Ali Tourani',
    python_requires=">=3.10",
    include_package_data=True,
    author="Ali Tourani, Yashar Deldjoo",
    author_email="a.tourani1991@gmail.com",
    url="https://github.com/RecSys-lab/Popcorn",
    long_description_content_type="text/markdown",
    long_description=open("README.md", encoding="utf-8").read(),
    description="🍿 A multi-faceted movie recommendation framework",
    packages=find_packages(include=["popcorn", "popcorn.*"], exclude=['docs', 'examples', 'rtd']),
    entry_points={
        'console_scripts': ['blenderproc=blenderproc.command_line:cli'],
    },
    install_requires=parseRequirements("requirements.txt"),
)
