from setuptools import setup

setup(
    name='pynominate',
    version='0.2',
    description='DW-NOMINATE in Python',
    author='Adam Boche, Jeff Lewis, Luke Sonnet',
    author_email='luke.sonnet@gmail.com',
    url='https://github.com/voteview/pynominate',
    packages=['pynominate', ],
    include_package_data=True,
    install_requires=["scikit-learn"],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
