import asyncio

from aio_pika.robust_connection import connect_robust

from rabbitmq.settings import RabbitmqData


class GenerationStorage:
    def __init__(
        self,
        ssl: bool = False,
        host: str = RabbitmqData.generation.host,
        port: int = RabbitmqData.generation.port,
        login: str = RabbitmqData.generation.login,
        password: str = RabbitmqData.generation.password,
        virtualhost: str = RabbitmqData.generation.virtualhost,
    ) -> None:
        self._ssl = ssl
        self._host = host
        self._port = port
        self._login = login
        self._channel = None
        self._connection = None
        self._password = password
        self._virtualhost = virtualhost

    async def connection(self):
        return await connect_robust(
            ssl=self._ssl,
            port=self._port,
            host=self._host,
            login=self._login,
            password=self._password,
            virtualhost=self._virtualhost,
            loop=asyncio.get_running_loop(),
        )


class StatusStorage:
    def __init__(
        self,
        ssl: bool = False,
        host: str = RabbitmqData.status.host,
        port: int = RabbitmqData.status.port,
        login: str = RabbitmqData.status.login,
        password: str = RabbitmqData.status.password,
        virtualhost: str = RabbitmqData.status.virtualhost,
    ) -> None:
        self._ssl = ssl
        self._host = host
        self._port = port
        self._login = login
        self._channel = None
        self._connection = None
        self._password = password
        self._virtualhost = virtualhost

    async def connection(self):
        return await connect_robust(
            ssl=self._ssl,
            port=self._port,
            host=self._host,
            login=self._login,
            password=self._password,
            virtualhost=self._virtualhost,
            loop=asyncio.get_running_loop(),
        )
