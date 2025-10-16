from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="music-recommendation-engine",
    version="1.0.0",
    description="Spotify-style music recommendation system with collaborative filtering and deep learning",
    author="Zheng Dong",  
    author_email="a13105129007@gmail.com", 
    url="https://github.com/zhengbrody/music-recommendation-engine",  
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
)