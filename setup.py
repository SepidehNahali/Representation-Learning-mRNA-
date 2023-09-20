from setuptools import setup, find_packages

setup(
    name="YourPackageName",  # Replace with your package name
    version="0.1",
    description="A description of your package",
    author="Your Name",
    author_email="your@email.com",
    url="https://github.com/yourusername/your-repo-name",  # Replace with your GitHub repository URL
    packages=find_packages(),
    install_requires=[
        "viennarna",  # Add any other dependencies here
        "RNA",
        "python-Levenshtein",
        "pandas",
        "numpy",
        "seaborn",
        "matplotlib",
        "tqdm",
        "biopython",
        "scikit-learn",
        "keras",
        "tensorflow",
        "tensorflow_addons",
    ],
)
