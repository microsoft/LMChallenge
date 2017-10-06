from setuptools import setup

with open('version.txt', 'r') as v:
    version = v.readline().strip()

with open('README.md', 'r') as r:
    readme = r.read()

setup(name='lmchallenge',
      version=version,
      description='LM Challenge'
      ' - A library & tools to evaluate predictive language models.',
      long_description=readme,

      url='https://github.com/Microsoft/LMChallenge',
      author='Microsoft Corporation',
      author_email='swiftkey-deep@service.microsoft.com',
      license='MIT',

      packages=['lmchallenge', 'lmchallenge.core'],
      install_requires=[
          'click',
          'emoji',
          'regex',
      ],
      tests_require=[
          'nose',
      ],
      test_suite='nose.collector',
      entry_points='''
      [console_scripts]
      lmc=lmchallenge:cli

      lmc-run=lmchallenge.run:cli
      lmc-stats=lmchallenge.stats:cli
      lmc-ic-opt=lmchallenge.ic_opt:cli
      lmc-pretty=lmchallenge.pretty:cli
      lmc-page=lmchallenge.page:cli
      ''')
