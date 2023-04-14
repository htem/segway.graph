from setuptools import setup

setup(
        name='segway.graph',
        version='1.0',
        url='https://github.com/htem/segway.graph',
        author='Tri Nguyen',
        author_email='tri_nguyen@hms.harvard.edu',
        license='MIT',
        packages=[
            'segway.graph'
        ],
        install_requires=[
            "networkx",
            "numpy",
            "jsmin",
        ]
)
