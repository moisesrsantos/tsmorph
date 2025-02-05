from setuptools import setup, find_packages

setup(
    name='tsmorph',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'pycatch22',
        'neuralforecast'
    ],
    author='Moisés Santos',
    description='A package for generating semi-synthetic time series using morphing techniques.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/tsmorph',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
