from setuptools import setup, find_packages

setup(
    name='c-GNSS',
    version='0.1.0',
    author='Ibaad Anwar',
    author_email='ibaadanwar20@iitk.ac.in',
    description='A package for the Characterization of a GNSS station',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/anwar1406/c-GNSS',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'seaborn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    package_data={
        '': ['*.csv', '*.txt', '*.xlsx'],
    },
    include_package_data=True,
)
