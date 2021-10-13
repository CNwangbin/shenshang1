#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['click>=6.0', 'numpy', 'scipy', 'pandas', 'matplotlib', 'joblib']

extras_requirements = {'progress_bar': ['tqdm']}

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Zech Xu",
    author_email='zhenjiang.xu@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="shenshang computes co-occurance and mutual exclusivity in microbiome data in a compositionality-insensitive manner.",
    entry_points={
        'console_scripts': [
            'shenshang=shenshang.cli:main',
        ],
    },
    install_requires=requirements,
    extras_requires=extras_requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='shenshang',
    name='shenshang',
    packages=find_packages(include=['shenshang']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/RNAer/shenshang',
    version='0.1.0',
    zip_safe=False,
)
