from setuptools import setup, find_packages

setup(
    name="python-minigrad",
    packages=find_packages("src"),
    url="https://github.com/avramdj/minigrad",
    license="MIT License",
    author="Avram Djordjevic",
    author_email="avramdjordjevic2@gmail.com",
    description="Automatic tensor differentiation engine with a deep learning library on top.",
)
