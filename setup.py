
import setuptools

with open('README.md', 'r', encoding='utf-8') as doc:
    long_description = doc.read()

setuptools.setup(
    name='trump_bot',
    version='0.1.1',
    author='Hakula Chen',
    author_email='i@hakula.xyz',
    description='A simple Twitter bot which tries to mimic @realDonaldTrump.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hakula139/Trump-bot',
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3.8',
        'Environment :: GPU :: NVIDIA CUDA :: 10.1',
        'Operating System :: OS Independent',
    ],
    python_requires='3.8',
)
