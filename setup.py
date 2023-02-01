import setuptools

# with open('requirements.txt') as f:
#     requirements = f.read().splitlines()

setuptools.setup(
    name="dynamicxs",
    version="0.0.1",
    author="ninarina12",
    author_email="",
    description="TBD",
    long_description="TBD",
    url="https://github.com/ninarina12/dynamiCXS",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=["numpy"]
)
