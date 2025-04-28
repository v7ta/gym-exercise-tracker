from setuptools import setup, find_packages

setup(
    name="gym_exercise_processor",
    version="0.1.0",
    description="A Python project for processing gym exercise videos.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "torch",
        "numpy",
        "mediapipe",
        "tqdm",
        "scikit-learn",
        "kagglehub",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "classifier_pipeline=gym_exercise_processor.scripts.classifier_pipeline:main",
            "process_video=gym_exercise_processor.scripts.process_video:main",
            "process_camera=gym_exercise_processor.scripts.process_camera:main",
        ],
    },
)