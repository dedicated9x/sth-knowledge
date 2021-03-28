class Number:
    def __init__(self, value, type_=None):
        self.value = value
        self.type_ = type_

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        str_ = self.value
        if self.type_ is not None:
            str_ = f"{self.type_}: {str_}"
        return str_

class Person:
    def __init__(self, name, surname):
        self.name = name
        self.surname = surname
        self.numbers = []

    def add_number(self, number):
        self.numbers.append(number)

    def __hash__(self):
        return hash((self.name, self.surname))

    def __repr__(self):
        return f"{self.name} {self.surname}: {', '.join([str(n) for n in self.numbers])}"


class Phonebook:
    def __init__(self):
        self._numbers_cache = {}
        self._people_cache = {}

    def add_number(self, name, surname, number, type=None):
        new_number = Number(number, type)
        new_person = Person(name, surname)
        new_person.add_number(new_number)
        self._numbers_cache[hash(new_number)] = new_number
        self._people_cache[hash(new_person)] = new_person

    def __repr__(self):
        return '\n'.join([str(obj) for obj in self._people_cache.values()])


book = Phonebook()
book.add_number('Ala', 'Wesołowska', '+048 513 056 121', 'main')
# book.add_number('Ala', 'Wesołowska', '22-848-34-21')
print(book)
# TODO nie dodajemy, jak kto inny ma
# TODO nie dodajemy drugi raz takiego samego
# Ala Wesołowska: +048 513 056 121, 22-848-34-21


"""test repra"""
# p = Person('Ala', 'Wesołowska')
# n1 = Number('+048 513 056 121', 'main')
# n2 = Number('22-848-34-21')
# p.add_number(n1)
# p.add_number(n2)
