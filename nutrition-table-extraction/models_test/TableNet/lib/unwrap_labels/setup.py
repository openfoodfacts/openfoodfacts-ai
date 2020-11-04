from setuptools import setup, find_packages


__author__ = 'Alexey Zankevich'


with open('requirements.txt') as f:
    requirements = f.read().strip()

setup(
    name="unwrap_labels",
    version="1.1.2",
    py_modules=['unwrap_labels'],
    author=__author__,
    author_email="alex.zankevich@gmail.com",
    description="Library to unwrap labels using OpenCV",
    keywords=["OpenCV"],
    license="MIT",
    platforms=['Platform Independent'],
    url="https://github.com/Nepherhotep/unwrap_labels",
    install_requires=requirements,
    classifiers=["Development Status :: 5 - Production/Stable",
                 "Programming Language :: Python :: 2.7",
                 "Programming Language :: Python :: 3.5",
                 "Programming Language :: Python :: 3.6",
                 "Programming Language :: Python :: 3.7"]
)
