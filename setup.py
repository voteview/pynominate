from setuptools import setup

setup(
    name='pynominate',
    version='0.1',
    description='DW-NOMINATE in Python',
    author='Jeff Lewis, Luke Sonnet',
    author_email='luke.sonnet@gmail.com',
    url='https://github.com/voteview/pynominate',
    packages=['pynominate', ],
    include_package_data=True,
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
