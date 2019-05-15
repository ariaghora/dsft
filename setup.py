from setuptools import setup

setup(
    name='dsft',
    version=version['__version__'],
    description=('Domain Specific Feature Transfer'),
    long_description='The implementation of domain specific feature transfer (DSFT) by Pengfei et al (2017), "A General Domain Specific Feature Transfer Framework for Hybrid Domain Adaptation" in IEEE Transactions on Knowledge and Data Engineering.',
    author='Aria Ghora Prabono',
    author_email='hello@ghora.net',
    url='https://github.com/ariaghora/dsft',
    license='MIT',
    packages=['dsft'],
#   no dependencies in this example
#   install_requires=[
#       'dependency==1.2.3',
#   ],
#   no scripts in this example
#   scripts=['bin/a-script'],
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.6'],
    )