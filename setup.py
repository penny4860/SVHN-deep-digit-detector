from distutils.core import setup

setup(name='number-detector',
      version=open("digit_detector/_version.py").readlines()[-1].split()[-1].strip("\"'"),
      description='SVHN number detector and recognizer',
      author='jeongjoonsup',
      author_email='penny4860@gmail.com',
      url='https://penny4860.github.io/',
      packages=['digit_detector'],  
     )

