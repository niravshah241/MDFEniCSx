# Pull dolfinx 0.7.3 docker image
FROM dolfinx/dolfinx:v0.7.3

# Set work directory
RUN mkdir -p /home/mdfenicsx
WORKDIR /home/mdfenicsx

# Install MDFEniCSx from github
RUN pip install git+https://github.com/niravshah241/MDFEniCSx.git

# Keep container alive
CMD ["sleep", "infinity"]
