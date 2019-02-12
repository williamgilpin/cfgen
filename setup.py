"""A setup module for portbalance."""

from setuptools import setup, find_packages
from setuptools.command.install import install as _install

# Adapted from https://stackoverflow.com/questions/26799894/installing-nltk-data-in-setup-py-script
# Subclasses default installer 
class Install(_install):
    def run(self):
        _install.do_egg_install(self)
        import nltk
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')

setup(
    name='cfgen',
    version='0.1.0',
    author='William Gilpin',
    author_email='wgilpin@stanford.edu',
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: OS Independent',
    ],
    description='Uses a combination of Markov chains and context-free-grammars to'
                +' generate random sentences with features of both language models.',
    keywords="finance portfolio rebalancing stock",
    python_requires='>=3',
    cmdclass={'install': Install},
    install_requires=[
        'numpy',
        'nltk' 
    ],
    setup_requires=['nltk'],
    include_package_data=True, 
    # dependency_links = ['git+https://github.com/emilmont/pyStatParser@master#egg=pyStatParser-0.0.1'],
    # dependency_links=['https://github.com/emilmont/pyStatParser/tarball/master#egg=pyStatParser-0.0.1'],
    dependency_links = ['git+https://github.com/emilmont/pyStatParser.git#egg=pyStatParser-0.0.1'],
    packages=["cfgen"],
    url='https://github.com/williamgilpin/cfgen'
)