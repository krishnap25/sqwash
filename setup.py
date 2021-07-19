import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sqwash",
    version="0.1.0",
    author="Krishna Pillutla",
    author_email="pillutla@cs.washington.edu",
    description="Distributionally Robust Learning with the Superquantile",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/krishnap25/sqwash",
    project_urls={
        "Bug Tracker": "https://github.com/krishnap25/sqwash/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'torch>=1.7.0',
    ]
)
