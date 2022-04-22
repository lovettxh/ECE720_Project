from setuptools import setup
# or
# from distutils.core import setup

setup(
        name='autotest',  
        version='1.0',  
        description='ECE720 project autotest system', 
        author='Hao Xuan, Yinsheng He', 
        author_email='hxuan@ualberta.ca, yinsheng@ualberta.ca',
        url='https://github.com/lovettxh/ECE720_Project', 
        include_package_data=True,
        packages=['autotest'], 
)