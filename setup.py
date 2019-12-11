from setuptools import setup, find_packages
from pkg_resources import resource_filename
from pathlib import Path


with open(resource_filename("eFFORT", "version.txt"), "r") as vf:
    version = vf.read().strip()

with (Path(__file__).parent / 'readme.md').open() as readme_file:
    readme = readme_file.read()

setup(
    name='eFFORT',
    packages=find_packages(),
    url='https://github.com/b2-hive/eFFORT',
    author='Markus Tobias Prim, Maximilian Welsch',
    author_email='markus.prim@kit.edu',
    description='A tool for convenient reweighting between different form '
                'factors of semileptonic B decays.',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'tabulate',
        'uncertainties',
        'numdifftools',
        'pandas',
        'click',
    ],
    include_package_data=True,
    long_description=readme,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License "
    ],
    license='MIT',
    entry_points='''
        [console_scripts]
        download_btodstarstarlnu_data=eFFORT.SLBToC.utility:download_botdstarstarlnu_data
    '''
)
