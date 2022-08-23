import os

import setuptools

here = os.path.abspath(os.path.dirname(__file__))

# Dependencies.
with open("docs/requirements.txt") as f:
    requirements = f.readlines()
install_requires = [t.strip() for t in requirements]

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()
    
#setuptools.setup(name='oceanliner')

setuptools.setup(
    name="oceanliner",
    author="Kayla Drushka",
    #author_email="ooipython@gmail.com",
    #description="A python toolbox for acquiring and analyzing Ocean Obvservatories Initiative (OOI) Data",
    #long_description=long_description,
    #long_description_content_type="text/markdown",
    #url="https://github.com/ooipy/ooipy",
    packages=setuptools.find_packages(exclude=("osse_tools_mb")),
    include_package_data=True,
    #package_data={"": ["hydrophone/*.csv"]},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    #py_modules=["_ooipy_version"],
    #use_scm_version={
    #    "write_to": "_ooipy_version.py",
       # "write_to_template": 'version = "{version}"\n',
       # "local_scheme": "no-local-version",
    #}
    )
    
