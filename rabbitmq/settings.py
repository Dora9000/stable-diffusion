import os
import settings


class RabbitmqData:
    class BaseData:
        login = settings.RABBITMQ_LOGIN
        host = settings.RABBITMQ_HOST
        password = settings.RABBITMQ_PASSWORD
        port = settings.RABBITMQ_PORT
        virtualhost = ""
        ssl = False

        default_queue = ""

    class GenerationData(BaseData):
        virtualhost = os.getenv("RABBITMQ_GENERATION_HOST", "generation")
        default_queue = os.getenv("RABBITMQ_GENERATION_QUEUE_NAME", "generation")

    class StatusData(BaseData):
        virtualhost = os.getenv("RABBITMQ_STATUS_HOST", "status")
        default_queue = os.getenv("RABBITMQ_STATUS_QUEUE_NAME", "status")

    generation = GenerationData()
    status = StatusData()
