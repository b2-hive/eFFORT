from setuptools import setup

setup(
    name='eFFORT',
    version='v.0.1.1',
    packages=['eFFORT'],
    url='https://gitlab.ekp.kit.edu/mapr/eFFORT',
    license='',
    author='Markus Prim',
    author_email='markus.prim@kit.edu',
    description='A tool for convenient reweighting between different form factors of semileptonic B decays.',
    install_requires=['numpy', 'scipy', 'matplotlib', 'tabulate', 'uncertainties', 'numdifftools', 'click'],
    entry_points='''
        [console_scripts]
        download_btodstarstarlnu_data=eFFORT.SLBToC.utility:download_botdstarstarlnu_data
    '''
)
