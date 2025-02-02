import asyncio

import grpc
import tyro

from cornserve.logging import get_logger
from cornserve.services.pb import (
    shard_manager_pb2,
    shard_manager_pb2_grpc,
    worker_pb2,
    worker_pb2_grpc,
    common_pb2,
)

logger = get_logger(__name__)
cleanup_coroutines = []


class ShardManagerServicer(shard_manager_pb2_grpc.ShardManagerServicer):
    """Shard Manager gRPC service implementation."""

    def __init__(self, workers: dict[str, str]) -> None:
        """Initialize the ShardManagerServicer.

        Args:
            workers: A dictionary of worker IDs and gRPC addresses.
        """
        self.worker_stubs: dict[str, worker_pb2_grpc.WorkerStub] = {
            worker_id: worker_pb2_grpc.WorkerStub(
                grpc.aio.insecure_channel(worker_address)
            )
            for worker_id, worker_address in workers.items()
        }

    async def Healthcheck(
        self,
        request: shard_manager_pb2.HealthcheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> shard_manager_pb2.HealthcheckResponse:
        """Recursively check and report the health of all workers."""
        resp = shard_manager_pb2.HealthcheckResponse()
        resp.status = common_pb2.Status.STATUS_OK
        for worker_id, worker_stub in self.worker_stubs.items():
            worker_resp: worker_pb2.HealthcheckResponse = worker_stub.Healthcheck(
                worker_pb2.HealthcheckRequest()
            )
            worker_status = shard_manager_pb2.WorkerStatus(
                worker_id=worker_id, status=worker_resp.status
            )
            resp.worker_statuses[worker_id] = worker_status
        return resp


async def serve(ip: str = "[::]", port: int = 50051) -> None:
    server = grpc.aio.server()
    shard_manager_pb2_grpc.add_ShardManagerServicer_to_server(
        ShardManagerServicer(), server
    )
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)

    await server.start()

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        # Shuts down the server with 5 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(5)
        logger.info("Server stopped")

    cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(tyro.cli(serve))
    finally:
        loop.run_until_complete(*cleanup_coroutines)
        loop.close()
