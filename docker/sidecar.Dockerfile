FROM python:3.12.8
RUN pip install torch torchvision torchaudio

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e .

ENTRYPOINT ["python", "-u", "-m", "cornserve.services.sidecar.server"]
