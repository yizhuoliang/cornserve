FROM python:3.12.8

ADD . /workspace/cornserve

WORKDIR /workspace/cornserve/python
RUN pip install -e .

ENTRYPOINT ["python", "-m", "cornserve.services.task_manager.server"]
