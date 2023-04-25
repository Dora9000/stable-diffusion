import json
import uuid

from aio_pika import Message

from rabbitmq.settings import RabbitmqData
from rabbitmq.storage import StatusStorage
from rabbitmq.utils import Singleton


class StatusProducer(metaclass=Singleton):
    def __init__(self) -> None:
        self.storage = None
        self.default_routing_key = RabbitmqData.status.default_queue

    def _data_wrapper(self, data: dict) -> dict:
        # message_id
        # generation_message_id
        # percent
        return {**data, "message_id": uuid.uuid4().hex}

    async def send(self, data: dict) -> None:
        message = json.dumps(self._data_wrapper(data)).encode("utf-8")

        if not self.storage:
            connection = await StatusStorage().connection()
            self.storage = await connection.channel()
            await self.storage.declare_queue(self.default_routing_key)

        await self.storage.default_exchange.publish(Message(body=message), routing_key=self.default_routing_key)

