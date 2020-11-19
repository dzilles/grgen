# Copyright (c) 2020 Daniel Zilles

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Setup 
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(

    name='grgen',

    version='0.0.1',

    description='A mesh generation tool using machine learning',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/dzilles/grgen',

    author='Daniel Zilles',

    author_email='daniel.zilles@rwth-aachen.de',

    license='MIT License',

    classifiers=[  
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

    keywords='grid, mesh, meshing, machine learning, Kohonen, som, self-organizing map',  

    packages=['grgen'],
    #packages=find_packages('grgen'),

    python_requires='>=3.5, <4',

    install_requires=['tensorflow', 'numpy<1.19.0,>=1.16.0', 'matplotlib', 'scipy', 'shapely', 'imageio'],  # Optional

    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },

    # package_data={ 
    #     'sample': ['package_data.dat'],
    # },

    # data_files=[('my_data', ['data/data_file'])],

    # entry_points={
    #     'console_scripts': [
    #         'grgen=grgen.grgen',
    #     ],
    # },

    project_urls={
        'Bug Reports': 'https://github.com/dzilles/grgen/issues',
    #    'Funding': '',
    #    'Say Thanks!': '',
        'Source': 'https://github.com/dzilles/grgen/',
    },
)
