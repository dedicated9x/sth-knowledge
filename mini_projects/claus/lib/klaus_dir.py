from lib.paths_registry import PathsRegistry

# TODO klausdirectory
class KlausDir:
    @staticmethod
    def _get_max_record_number():
        path_to_records = PathsRegistry.records
        return max(
            [int(path.stem) for path in path_to_records.glob('**/*') if path.is_file()]
        )

    @classmethod
    def get_next_available_record_filename(cls):
        max_record_number = cls._get_max_record_number()
        available_filename = f"{max_record_number + 1}.mp3"
        return available_filename
