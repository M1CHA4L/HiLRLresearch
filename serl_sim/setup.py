from setuptools import setup, find_packages

setup(
    name="serl_sim",
    version="0.1.0",
    description="MuJoCo UR7e simulation + simulated human for HIL-SERL attention research",
    packages=find_packages(),
    package_data={"serl_sim": ["assets/*.xml"]},
    include_package_data=True,
    install_requires=["mujoco>=3.1", "gymnasium", "numpy", "scipy"],
    python_requires=">=3.10",
)
