from redis import Redis
from rq import Queue, Connection

# Redis connection
redis_conn = Redis(host="localhost", port=6379, db=0)

# Default queue
queue = Queue("autored", connection=redis_conn)
