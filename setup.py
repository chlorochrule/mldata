# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='mldata',
    version='0.1.0',
    description='Machine Learning Database and data loader',
    long_description=readme,
    author='Naoto MINAMI',
    author_email='minami.polly@gmail.com',
    install_requires=['numpy', 'scikit-learn', 'pillow', 'tensorflow'],
    url='',
    license=license,
    packages=find_packages(exclude=('tests', 'examples'))
)
