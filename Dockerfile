FROM python:3.5

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

ADD . /tmp/lmc
RUN cd /tmp/lmc/                    \
    && python3 setup.py install     \
    && rm -r /tmp/lmc
