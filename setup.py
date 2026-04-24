from setuptools import setup, find_packages

setup(
    name="default_risk",
    version="0.1.0",
    # IMPORTANTE: Esto mapea la raíz del código a la carpeta src
    package_dir={"": "src"},
    # Esto busca cualquier carpeta con un __init__.py dentro de src
    packages=find_packages(where="src"),
)