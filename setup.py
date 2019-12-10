from setuptools import setup, find_packages

setup(
    name='eFFORT',
    version='v.0.2.0',
    packages=find_packages(),
    url='https://gitlab.ekp.kit.edu/mapr/eFFORT',
    license='',
    author='Markus Tobias Prim, Maximilian Welsch',
    author_email='markus.prim@kit.edu',
    description='A tool for convenient reweighting between different form factors of semileptonic B decays.',
    install_requires=['numpy', 'scipy', 'matplotlib', 'tabulate', 'uncertainties', 'numdifftools', 'pandas'],
)
