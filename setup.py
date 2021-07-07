#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [s.strip() for s in open('requirements.txt', 'r').readlines()] 

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Guillermo E. Blanco",
    author_email='geblanco@lsi.uned.es',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description="Utils to run multiple choice question answering with huggingface transformers",
    entry_points={
        'console_scripts': [
            'mc_transformers=mc_transformers.mc_transformers:main',
            'mc_transformers_windowing=mc_transformers.window_examples:main',
        ],
    },
    extras_require={
        'windowing': ['nltk']
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='mc_transformers',
    name='mc_transformers',
    packages=find_packages(include=['mc_transformers', 'mc_transformers.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/geblanco/mc_transformers',
    version='0.1.12',
    zip_safe=False,
)
