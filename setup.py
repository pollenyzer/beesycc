import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="beesycc",
    version="0.0.1",
    author="Parzival",
    author_email="app@trio.li",
    install_requires=[],
    description=(
        "Tools for using everyday objects as color calibration references."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["beesycc"]),
    python_requires=">=3.6",
)
