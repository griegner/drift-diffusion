# hash:sha256:56fd9361dbddae3ba63e197d400d330ac2e5007a9d68634ba1fc92ad6721721c
FROM registry.codeocean.com/codeocean/py-r:python3.12.8-R4.4.2-JupyterLab4.3.5-RStudiorstudio-server-2024.12.1-563-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

COPY postInstall /
RUN /postInstall
