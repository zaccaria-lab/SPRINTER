import setuptools
from setuptools import setup


setuptools.setup(
    name='sprinter',
    version='1.0',
    python_requires='>=3',
    packages=['sprinter', 'sprinter.bin', 'sprinter.libs'],
    author='Simone Zaccaria and Olivia Lucas',
    author_email='s.zaccaria@ucl.ac.uk',
    description='Single-cell Proliferation Rate Inference in Non-homogeneous Tumours through Evolutionary Routes (SPRINTER)',
    long_description='https://github.com/zaccaria-lab/sprinter',
    url='https://github.com/zaccaria-lab/sprinter',
    install_requires=[
        'importlib_resources; python_version < "3.9"',
        'numpy<=1.26.4',
        'pandas',
        'scipy',
        'statsmodels',
        'hmmlearn',
        'matplotlib',
        'pybedtools',
        'scikit-learn',
        'seaborn',
        'numba'
    ],
    license='Custom based on BSD 3-clause extension',
    platforms=["Linux", "MacOs", "Windows"],
    classifiers=[
        'Programming Language :: Python :: 3',
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    include_package_data=True,
    package_data={'sprinter' :['sprinter/resources/*']},
    zip_safe=True,
    keywords=[
        'scientific',
        'sequence analysis',
        'cancer',
        'single-cell',
        'DNA',
        'copy-number',
        'proliferation'],
    entry_points={'console_scripts': ['sprinter=sprinter.bin.sprinter:main',]}
)
