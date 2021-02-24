
class KlausDir:
    def __init__(self, path_to_klaus_dir):
        self.path_to_records = path_to_klaus_dir.joinpath("db", "niem_60", "wav")

    def _get_max_record_number(self):
        return max(
            [int(path.stem) for path in self.path_to_records.glob('**/*') if path.is_file()]
        )

    def get_next_available_record_filename(self):
        max_record_number = self._get_max_record_number()
        available_filename = f"{max_record_number + 1}.mp3"
        return available_filename
