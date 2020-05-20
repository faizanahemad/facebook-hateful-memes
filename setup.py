from setuptools import find_packages, setup

setup(name='facebook-hateful-memes',
      version='0.0.1',
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
      keywords=['Pandas','numpy','data-science','IPython', 'Jupyter','ML','Machine Learning'],
      packages=find_packages(),
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
