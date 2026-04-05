from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time, os, requests

DOCS_DIR = "/app/docs"
INGEST_ENDPOINT = "http://127.0.0.1:8088/ingest"

class DocHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith(".pdf"):
            print(f"[WATCHER] Nuevo PDF: {event.src_path}")
            with open(event.src_path, "rb") as f:
                requests.post(
                    INGEST_ENDPOINT,
                    files={"file": (os.path.basename(event.src_path), f)}
                )

def start_watcher():
    observer = Observer()
    observer.schedule(DocHandler(), DOCS_DIR, recursive=False)
    observer.start()
    print(f"[WATCHER