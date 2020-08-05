from setuptools import setup, find_packages

setup(
    name='plumes',
    version='0.0.1',
    license='GPL v3',
    description='ice shelf buoyant plume models',
    author='Daniel Shapero',
    url='https://github.com/icepack/plumes',
    packages=find_packages(exclude=['doc', 'test']),
    install_requires=['numpy', 'scipy', 'firedrake'],
    extras_require={
        'doc': ['sphinx', 'sphinxcontrib-bibtex', 'sphinx_rtd_theme',
                'ipykernel', 'nbconvert', 'matplotlib', 'tqdm']
    }
)
