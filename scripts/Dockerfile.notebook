FROM jupyter/scipy-notebook

USER root
ADD . /home/jovyan/lmchallenge
RUN cd /home/jovyan/lmchallenge             \
    && python3 setup.py install             \
    && pip3 install -r requirements-dev.txt

USER jovyan
