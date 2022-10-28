import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Dependencies.
with open("requirements.txt") as f:
    requirements = f.readlines()
install_requires = [t.strip() for t in requirements]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    
#setuptools.setup(name='oceanliner')

setuptools.setup(
    name="Oceanliner",
    author="Kyla Drushka",
    #author_email="",
    #description="A python toolbox for analyzing NASA Surface Water Ocean Topography (SWOT) Satellite Data for planning in-situ research campaigns",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="",
    #packages=setuptools.find_packages(exclude=("osse_tools_mb")),
    include_package_data=True,
    #package_data={"": ["hydrophone/*.csv"]},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    py_modules=["oceanliner"],
    
    )
    
