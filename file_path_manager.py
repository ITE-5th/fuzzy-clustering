import glob
import os


class FilePathManager:
    root_path = os.path.dirname(os.path.abspath(__file__)) + "/"

    @staticmethod
    def resolve(path: str):
        return FilePathManager.root_path + path

    @staticmethod
    def clear_dir(path: str):
        path = FilePathManager.resolve(path + "/*")
        files = glob.glob(path)
        for f in files:
            os.remove(f)
