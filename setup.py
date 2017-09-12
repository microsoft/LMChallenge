from setuptools import setup

with open('version.txt', 'r') as v:
    version = v.readline().strip()

setup(name='lmchallenge',
      version=version,
      description='LM Challenge - language modelling evaluation suite',
      url='https://github.com/TouchType/LMChallenge',
      packages=['lmchallenge', 'lmchallenge.core'],
      install_requires=[
          'click',
          'emoji',
          'regex',
      ],
      tests_require=[
          'nose',
      ],
      package_data={'lmchallenge': ['data/*']},
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
