import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch-hd", # Replace with your own username
    version="0.0.0.2",
    author="Rishikanth",
    author_email="r3chandr@ucsd.edu",
    description="Optimized implementations of HD functions using pytorch with GPU support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://rishikanthc.github.io/torch-hd/",
    project_urls={
        "Bug Tracker": "https://github.com/rishikanthc/torch-hd/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(exclude=['tests']),
    python_requires=">=3.6",
)
