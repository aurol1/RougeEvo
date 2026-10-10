FROM python:3.11-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends libreoffice-calc-nogui fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir openpyxl==3.1.5 pandas==2.3.3 numpy==2.2.6

COPY docker/registrymodifications.xcu /opt/rouge-lo/user/registrymodifications.xcu
RUN HOME=/tmp soffice -env:UserInstallation=file:///opt/rouge-lo --headless --terminate_after_init \
    && grep -q 'Formula/Load"><prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0<' \
         /opt/rouge-lo/user/registrymodifications.xcu \
    && chmod -R a+rX /opt/rouge-lo \
    && libreoffice --version

ENV HOME=/tmp PYTHONDONTWRITEBYTECODE=1
WORKDIR /work
USER 65534:65534
