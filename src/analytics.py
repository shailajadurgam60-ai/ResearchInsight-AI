import time


class SessionAnalytics:
    """Tracks usage statistics for the current session."""

    def __init__(self):
        self.total_queries = 0
        self.total_docs = 0
        self.total_chunks = 0
        self.total_pages = 0
        self.response_times: list[float] = []
        self._query_start_time: float = 0.0

    def record_documents(self, num_docs: int, num_pages: int, num_chunks: int):
        self.total_docs = num_docs
        self.total_pages = num_pages
        self.total_chunks = num_chunks

    def start_query(self):
        self._query_start_time = time.time()

    def end_query(self):
        elapsed = time.time() - self._query_start_time
        self.total_queries += 1
        self.response_times.append(elapsed)

    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)

    def last_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return self.response_times[-1]

    def summary(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "total_docs": self.total_docs,
            "total_pages": self.total_pages,
            "total_chunks": self.total_chunks,
            "avg_response_time": round(self.avg_response_time(), 2),
            "last_response_time": round(self.last_response_time(), 2),
        }
