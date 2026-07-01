"""
EMN: Epistemic Memory Networks
"""

from setuptools import setup, find_packages

setup(
    name="emn",
    version="0.1.0",
    author="Faaz Mohamed",
    description=(
        "Epistemic Memory Networks: treating memory confidence as a first-class "
        "architectural variable using Dirichlet evidential uncertainty."
    ),
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/FuzzDOT/emn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0",
        "transformers>=4.38",
        "accelerate>=0.28",
        "numpy>=1.24",
        "scipy>=1.11",
        "pandas>=2.0",
        "matplotlib>=3.7",
        "seaborn>=0.13",
        "scikit-learn>=1.3",
        "sentence-transformers>=2.6",
        "pydantic>=2.0",
        "anthropic>=0.25",
        "pyyaml>=6.0",
        "tqdm>=4.66",
    ],
    extras_require={
        "faiss": ["faiss-cpu>=1.7"],
        "wandb": ["wandb>=0.16"],
        "hydra": ["hydra-core>=1.3"],
        "avalanche": ["avalanche-lib>=0.4"],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=4.0",
            "black>=24.0",
            "isort>=5.13",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
