from setuptools import find_packages, setup

package_name = 'click_to_follow'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='f-noble',
    maintainer_email='136553218+f-noble@users.noreply.github.com',
    description='Displays an image and commands a neato to follow an object the user clicks',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'click_to_follow = click_to_follow.click_to_follow:main'
        ],
    },
)
