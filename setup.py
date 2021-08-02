#from distutils.core import setup
#from Cython.Build import cythonize
from setuptools import setup
from setuptools import Extension
import numpy

import setuptools

setup(  
        name = 'LANfactory',
        version='0.0.1',
        author = 'Alexander Fenger',
        url = 'https://github.com/AlexanderFengler/LANfactory',
        packages= ['lanfactory', 'lanfactory.config', 'lanfactory.trainers', 'lanfactory.utils'], # , 'ssms.basic_simulators', 'ssms.config', 'ssms.dataset_generators', 'ssms.support_utils'],
        description='Package with convenience functions to train LANs',
        install_requires= ['NumPy >= 1.17.0', 'SciPy >= 1.6.3', 'pandas >= 1.2.4', 'tensorflow >= 1.15'],
        setup_requires= ['NumPy >= 1.17.0', 'SciPy >= 1.6.3', 'pandas >= 1.2.4', 'tensorflow >= 1.15'],
        include_dirs = [numpy.get_include()] ,
        classifiers=[ 'Development Status :: 1 - Planning', 
                      'Environment :: Console',
                      'License :: OSI Approved :: MIT License',
                      'Programming Language :: Python',
                      'Topic :: Scientific/Engineering'
                    ]

    )


# package_data={'hddm':['examples/*.csv', 'examples/*.conf', 'keras_models/*.h5', 'cnn_models/*/*', 'simulators/*']},
# scripts=['scripts/hddm_demo.py'],