"""
Setup 
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='grgen',  # Required

    version='0.0.1',  # Required

    description='A mesh generation tool using machine learning',  # Optional

    long_description=long_description,  # Optional

    long_description_content_type='text/markdown',  # Optional

    url='https://github.com/dzilles/grgen',  # Optional

    author='Daniel Zilles',  # Optional

    author_email='daniel.zilles@rwth-aachen.de',  # Optional

    classifiers=[  # Optional
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],

    keywords='grid, mesh, meshing, machine learning, Kohonen, som, self-organizing map',  # Optional

    packages=['grgen'],#find_packages(),  # Required
    #packages=find_packages('grgen'),

    python_requires='>=3.5, <4',

    install_requires=['tensorflow', 'numpy<1.19.0,>=1.16.0', 'matplotlib', 'scipy', 'shapely', 'imageio'],  # Optional

    #extras_require={  # Optional
    #    'dev': ['check-manifest'],
    #    'test': ['coverage'],
    #},

    #package_data={  # Optional
    #    'sample': ['package_data.dat'],
    #},

    # data_files=[('my_data', ['data/data_file'])],  # Optional

    #entry_points={  # Optional
    #    'console_scripts': [
    #        'grgen=grgen.grgen',
    #    ],
    #},

    project_urls={  # Optional
        'Bug Reports': 'https://github.com/dzilles/grgen/issues',
    #    'Funding': '',
    #    'Say Thanks!': '',
        'Source': 'https://github.com/dzilles/grgen/',
    },
)
