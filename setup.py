try:
    from setuptools import setup
except ImportError:
    raise RuntimeError('setuptools is required')

import versioneer

KEYWORDS = [
    'photovoltaic',
    'solar',
    'analytics',
    'analysis',
    'performance',
    'PV'
    'satellite'
]


INSTALL_REQUIRES = [
    'scikit_image>=0.15.0',
    'matplotlib>=3.1.2',
    'requests>=2.22.0',
    'psycopg2_binary>=2.8.3',
    'opencv_python_headless>=4.1.2.30',
    'pandas>=0.25.1',
    'tensorflow>=2.1.0',
    'pytz>=2019.3',
    'numpy>=1.18.1',
    'boto3>=1.12.49',
    'botocore>=1.15.49',
    'Pillow>=7.2.0',
    'psycopg2>=2.8.5',
    'scikit-image>=0.17.2',
    'sklearn',
]

TESTS_REQUIRE = [
    'pytest >= 3.6.3',
]

EXTRAS_REQUIRE = {
    'doc': [
        'sphinx==1.8.5',
        'sphinx_rtd_theme==0.4.3',
        'ipython',
    ],
    'test': TESTS_REQUIRE
}

setup(
    name='panel_segmentation',
    version=versioneer.get_version(),
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    extras_require=EXTRAS_REQUIRE,
    description='A package to segment solar panels from a satellite image and perform automated metadata extraction.',
    url='https://github.com/NREL/Panel-Segmentation',
    keywords=KEYWORDS,
    author='Ayobami Edun, Kirsten Perry',
    author_email='aedun@ufl.edu; kirsten.perry@nrel.gov',
    license='MIT',
    packages=['panel_segmentation'],
    zip_safe=False
)
