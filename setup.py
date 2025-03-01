from setuptools import setup, find_packages

setup(
    name="gait_analysis_app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'mediapipe',
        'opencv-python',
    ],
)