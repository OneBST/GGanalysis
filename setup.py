import setuptools

NAME = 'GGanalysisLite'
DESCRIPTION = 'A simple and efficient computing package for Genshin Impact gacha analysis.'
URL = 'https://github.com/OneBST/GGanalysisLite'
EMAIL = 'onebst@foxmail.com'
AUTHOR = 'OneBST'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.2.2'
REQUIRED = [
    'numpy', 'scipy', 'matplotlib'
]

setuptools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(include=["GGanalysisLite*"]),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
)