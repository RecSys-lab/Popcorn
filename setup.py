from setuptools import setup, find_packages

setup(
    name="Popcorn",
    version="1.0.0",
    maintainer='Ali Tourani',
    author="Ali Tourani, Yashar Deldjoo",
    author_email="a.tourani1991@gmail.com",
    url="https://github.com/RecSys-lab/Popcorn",
    long_description_content_type="text/markdown",
    long_description=open("README.md", encoding="utf-8").read(),
    description="🍿 A multi-faceted movie recommendation framework",
    packages=find_packages(include=["popcorn", "popcorn.*"], exclude=['docs', 'examples', 'rtd']),
    include_package_data=True,
    python_requires=">=3.10",
    entry_points={
        'console_scripts': ['blenderproc=blenderproc.command_line:cli'],
    },
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.26",
        "opencv-python>=4.9",
        "matplotlib>=3.9",
        "pytube>=15.0",
        "scipy>=1.14.1",
        "requests>=2.32",
        "PyYAML>=6.0.1",
        "openai>=0.27.0",
        "scikit-learn>=1.6.1",
        "tensorflow>=2.17.0"
    ]
)
