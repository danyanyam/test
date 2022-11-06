from setuptools import setup, find_packages

setup(
    name='solution',
    version='0.0.1',
    author_email='dvbuchko@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    py_modules=['dvsolution'],
    install_requires=[
        'Click',
    ],
    entry_points={
        'console_scripts': [
            'dvsolution = dvsolution.scripts.main:cli',
        ],
    },
)

