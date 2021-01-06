FROM jupyter/datascience-notebook:latest
ENV JUPYTER_ENABLE_LAB=yes
RUN pip install numpy sklearn torch pandas matplotlib altair