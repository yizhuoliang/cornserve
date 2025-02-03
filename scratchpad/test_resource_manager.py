import asyncio

import grpc

from python.cornserve.services.pb import common_pb2, resource_manager_pb2, resource_manager_pb2_grpc

async def main():
    async with grpc.aio.insecure_channel("localhost:30001") as channel:
        stub = resource_manager_pb2_grpc.ResourceManagerStub(channel)
        request = resource_manager_pb2.DeployTaskManagerRequest(
            task_manager_id="the-best",
            type=common_pb2.TaskType.LLM,
            config={"model-id": "meta-llama/Llama-3.1-8B-Instruct"},
        )
        response = await stub.DeployTaskManager(request)
        print(response)

if __name__ == '__main__':
    asyncio.run(main())
