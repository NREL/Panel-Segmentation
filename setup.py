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
    'opencv_python_headless==4.5.2.52',
    'numpy>=1.18.1',
    'scikit_image>=0.16.2',
    'matplotlib>=3.1.3',
    'requests>=2.22.0',
    'tensorflow>=2.2.0',
    'pandas>=1.1.4',
    'cx_Freeze>=6.4.2',
    'Pillow>=8.0.1',
    'scikit_learn>=0.23.2',
    'h5py>=2.10.0',
    'detecto>=1.2.1'
]

TESTS_REQUIRE = [
    'pytest>=5.3.5'
]

EXTRAS_REQUIRE = {
    'doc': [
        'sphinx==1.8.5',
        'sphinx_rtd_theme>=0.5.0',
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
    description='A package to segment solar panels from a '
                'satellite image and perform automated metadata extraction.',
    url='https://github.com/NREL/Panel-Segmentation',
    keywords=KEYWORDS,
    author='Ayobami Edun, Kirsten Perry, Kevin Anderson, Christopher Campos',
    author_email='aedun@ufl.edu; kirsten.perry@nrel.gov; '
                'kevin.anderson@nrel.gov; chris.acampos@yahoo.com',
    package_data={
        'panel_segmentation': [
            'panel_segmentation/VGG16_classification_model.h5',
            'panel_segmentation/VGG16Net_ConvTranpose_complete.h5',
            'panel_segmentation/object_detection_model.pth',
            'panel_segmentation/examples/*',
            'panel_segmentation/tests/*',
        ],
    },
    include_package_data=True,
    license='MIT',
    packages=['panel_segmentation'],
    zip_safe=False
)
