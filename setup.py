from setuptools import setup, find_packages

setup(
    name='dac',
    version='0.1.0',
    description='Entity linker for Dutch historical newspapers.',
    long_description='Entity linker for Dutch historical newspapers.',
    url='https://github.com/jlonij/dac',
    author='Juliette Lonij',
    author_email='juliette.lonij@kb.nl',
    license='GPLv3+',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 \
            or later (GPLv3+)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7'
        ],
    packages=find_packages(where='.', exclude=['docs', 'tests']),
    install_requires=[
        'bottle', 'h5py', 'Keras', 'lxml', 'numpy', 'pandas',
        'python-Levenshtein', 'requests', 'scikit-learn', 'scipy', 'segtok',
        'tensorflow', 'Unidecode'
        ],
    package_data={'dac': [
        'config.json', 'features/bnn.json', 'features/features.json',
        'features/nn.json', 'features/svm.json', 'models/bnn.h5',
        'models/nn.h5', 'models/svm.pkl'
        ]},
    data_files=None,
    entry_points={}
)
