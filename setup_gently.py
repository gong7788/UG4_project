from distutils.core import setup

setup(name='CorrectingAgent',
      version='0.1dev',
      packages=['correctingagent',],
      license='Creative Commons Attribution-Noncommercial-Share Alike license',
      long_description=open('README.txt').read(), requires=['numpy', 'scipy'])
