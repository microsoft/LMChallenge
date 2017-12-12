from setuptools import setup

with open('version.txt', 'r') as v:
    version = v.readline().strip()

with open('README.md', 'r') as r:
    readme = r.read()

with open('requirements.txt', 'r') as r:
    requirements = list(x.strip() for x in r)

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
      install_requires=requirements,
      entry_points='''
      [console_scripts]
      lmc=lmchallenge:cli

      lmc-diff=lmchallenge.diff:cli
      lmc-grep=lmchallenge.grep:cli
      lmc-pretty=lmchallenge.pretty:cli
      lmc-run=lmchallenge.run:cli
      lmc-stats=lmchallenge.stats:cli
      lmc-validate=lmchallenge.validate:cli
      ''')
