from setuptools import setup, find_packages

setup(
    name='realesrgan-upscaler-gui',
    version='2.0.0',
    description='PyQt6-based RealESRGAN image upscaler with NVIDIA/AMD GPU support',
    long_description=open('README.md', encoding='utf-8').read() if __import__('os').path.isfile('README.md') else '',
    long_description_content_type='text/markdown',
    author='mikelongjr',
    url='https://github.com/mikelongjr/randotools',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        # GUI
        'PyQt6>=6.4.0',
        # Image processing
        'Pillow>=9.0.0',
        'opencv-python>=4.7.0',
        # RealESRGAN model
        'py_real_esrgan>=0.0.3',
        # Progress display
        'tqdm>=4.64.0',
        # PyTorch (user must install appropriate variant separately)
        # NVIDIA:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        # AMD:     pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
        # CPU:     pip install torch torchvision
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-qt>=4.2',
        ],
        'basicsr': [
            # Install build tools first: sudo dnf install gcc-c++ python3-devel
            'basicsr>=1.4.2',
        ],
    },
    entry_points={
        'console_scripts': [
            'realesrgan-upscaler-gui = upscaler.__main__:main',
        ],
        'gui_scripts': [
            'realesrgan-upscaler = upscaler.__main__:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: X11 Applications :: Qt',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Multimedia :: Graphics',
    ],
)