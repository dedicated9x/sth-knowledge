from lib.paths_registry import PathsRegistry


class KlausStateAnalyzer:
    @staticmethod
    def get_max_record_number():
        path_to_records = PathsRegistry.records
        return max(
            [int(path.stem) for path in path_to_records.glob('**/*') if path.is_file()]
        )
