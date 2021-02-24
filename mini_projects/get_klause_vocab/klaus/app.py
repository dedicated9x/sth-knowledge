from screeninfo import get_monitors
from dataclasses import dataclass

@dataclass
class LocationSet:
    explore_db_form: tuple
    explore_db_button: tuple
# TODO obiekty LocationSet sa na poczatku (widac, co jest grane)
# TODO LocationSet + AppInfo

class AppInfo:
    @staticmethod
    def get_location_set():
        monitors = get_monitors()
        if len(monitors) == 3:
            return LocationSet(
                explore_db_form=(822, 452),
                explore_db_button=(1382, 706)
            )
        else:
            return LocationSet(
                explore_db_form=(392, 201),
                explore_db_button=(1087, 520)
            )