from setuptools import find_packages, setup
from typing import List


HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]: 
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [requirement.replace('\n', '') for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements


setup(
    name='mlproject-1',
    version='0.0.1',
    author='Alexander Burgos',
    author_email='a.burgos.n05@gmail.com',
    packages=find_packages(), # look for all files with __init__.py
    install_requires = get_requirements('requirements.txt'),
)

