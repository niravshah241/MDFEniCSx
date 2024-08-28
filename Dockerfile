FROM dolfinx/dolfinx:v0.7.3

RUN mkdir -p /home/app
WORKDIR /home/app

COPY demo . 

RUN pip install git+https://github.com/niravshah241/MDFEniCSx.git

# ENTRYPOINT ["echo", "Hello-world"]
