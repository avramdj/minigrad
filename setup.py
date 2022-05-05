from setuptools import setup, find_packages

setup(
    name='minigrad',
    version='0.0.1',
    packages=find_packages('src'),
    url='https://github.com/avramdj/minigrad',
    license='MIT License',
    author='Avram Djordjevic',
    author_email='avramdjordjevic2@gmail.com',
    description='Automatic tensor differentiation engine with a deep learning library on top.'
)
