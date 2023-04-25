import os


class RabbitmqData:
    class BaseData:
        login = os.getenv("RABBITMQ_LOGIN", "admin")
        host = os.getenv("RABBITMQ_HOST", "194.26.196.6")
        password = os.getenv("RABBITMQ_PASSWORD", "password")
        port = 55075
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
