from setuptools import setup, find_packages

setup(
    name='realesrgan-upscaler-gui',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'realesrgan-upscaler-gui = upscaler.__main__:main',
        ],
    },
    install_requires=[
        'some_dependency',  # Replace with actual dependencies
    ],
    python_requires='>=3.6',
)