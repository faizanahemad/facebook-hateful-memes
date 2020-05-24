from setuptools import find_packages, setup

setup(name='facebook_hateful_memes_detector',
      version='0.0.2',
      description='',
      url='https://github.com/faizanahemad/facebook-hateful-memes',
      author='Faizan Ahemad',
      author_email='fahemad3@gmail.com',
      license='MIT',
      install_requires=[
          'numpy','pandas', 'more-itertools',
            'dill', 'seaborn','gensim', 'nltk',
            'joblib', 'opencv-python',
      ],
      keywords=[],
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=True)
