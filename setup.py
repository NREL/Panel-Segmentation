try:
    from setuptools import setup
except ImportError:
    raise RuntimeError('setuptools is required')
    
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
    'botocore>=1.16.14',
    'Pillow>=7.2.0',
    'psycopg2>=2.8.5',
    'skimage>=0.0'
]


setup(
    name='panel_segmentation',
    version='0.0.1',
    description='A package to segment solar panels from a satellite image and perform automated metadata extraction.',
    url='git@github.com:rfschubert/ptolemaios-sdk-package.git',
    keywords=KEYWORDS,
    author='Ayobami Edun, Kirsten Perry',
    author_email='aedun@ufl.edu; kirsten.perry@nrel.gov',
    license='unlicense',
    packages=['panel_segmentation'],
    zip_safe=False
)







