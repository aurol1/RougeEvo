from pathlib import Path
import shutil
import uuid


class TemporaryDirectory:
    def __init__(self):
        self.root = Path(__file__).resolve().parents[1] / '.test_tmp'
        self.root.mkdir(exist_ok=True)
        self.path = self.root / uuid.uuid4().hex
        self.path.mkdir()
        self.name = str(self.path)

    def __enter__(self):
        return self.name

    def __exit__(self, *_):
        self.cleanup()

    def cleanup(self):
        self.path.resolve().relative_to(self.root.resolve())
        if self.path.exists():
            shutil.rmtree(self.path)
