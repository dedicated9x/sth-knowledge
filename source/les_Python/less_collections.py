"""###################################################
        _____ _______ ______
 |        |   |______    |
 |_____ __|__ ______|    |
###################################################"""
''' enumerate zamiast range()'''
list_ = ['one', 'two', 'three']
_ = [f'{e} = {idx}' for idx, e in enumerate(list_)]

# TODO znaleźć lepszy przykład na enumerate

''' Slicing'''
# pyString = 'Python'
# sObject = slice(3)
# print(pyString[sObject])
# # #-> Pyt
#
# pyString = '1234567890'
# print(pyString[slice(1,6,3)])
# #-> 25
#
# pyString = 'Python'
# print(pyString[:2])
# #-> Py
#
# pyString = 'Python'
# print(pyString[2:])
# #-> thon
#
# pyString = 'Python'
# print(pyString[::-1])
# #-> nohTYp



"""###################################################
 ______  _____ _______ _______
 |    \\   |   |          |   
 |_____/ __|__ |_____     |   
###################################################"""
# TODO na co to komu? (deque)
"""###########################
DEQUE
##########################"""
# from collections import namedtuple
#
#
# Animal = namedtuple('Animal', 'name age type')
# perry = Animal(name="perry", age=31, type="cat")
# print(perry)
#
# '''
# Minus ->                                                                                                       immutable
# '''
#
# perry = perry._asdict()
# print(perry)
#
# '''
# Namedtuple lepsze od slownika, bo: (2)                                                                        1) szybsze                                                                                           2) latwiejszy dostep do pola?
# '''

''' list of tuples jako rozbudowany slownik'''
# tuple1 = ("pawel",25,"M")
# tuple2 = ("maks", 22, "K")
#
# list_of_tuples = [tuple1, tuple2]
#
# for name, age, gender in list_of_tuples:
#     print(name, age, gender)

'''comprehension'''
# a = {n: n**2 for n in range(5)}
# #NIE!:
# a = dict(n: n**2 for n in range(5))

'''#############################'''
''' kolejny sposob dodanie do slownika'''
'''#############################'''
# adict = {"ala":4, "ma":5, "kota":6}
# bdict = {"szla":7, "baba":8, "po":9, "lodzie":10}
# adict.update(bdict)
# print(adict)


''' kwestia rozpokowywania dictow - cdn...'''
# def func_(arg1, arg2):
#     print(arg1+arg2)
#
#
# # func_(arg1=2, arg2=3, arg3=4)
#
# dict1 = {"arg1":2, "arg2":3}
# dict2 = {"arg1":2, "arg2":3}
#
# # func_(dict1)
# func_(**dict2)


"""###################################################
 ______  _______  _____  _     _ _______
 |    \\ |______ |   __| |     | |______
 |_____/ |______ |___\\| |_____| |______ 
###################################################"""
# from collections import deque
#
# a = deque([4,5,6])
#
# a.popleft()
# a.append(7)
#
# print(a)
# if a:
#     print("puste")
#
# a.popleft()
# a.popleft()
# a.popleft()
#
# if a:
#     print("puste2")
#
#
# stack = [4,5,6]
# stack.pop()
# stack.append(7)
# print(stack)

"""http://www.network-science.de/ascii/         ->slant"""
"""#################################################
    _______   ____  ____  ___
   / ____/ | / / / / /  |/  /
  / __/ /  |/ / / / / /|_/ / 
 / /___/ /|  / /_/ / /  / /  
/_____/_/ |_/\____/_/  /_/   
#################################################"""
#TODO czy enumy maja jakies korzysci?

# from enum import Enum as _Enum
#
#
# class ScopeLevels(_Enum):
#     """
#     Class for helping with typo mistakes.
#
#     """
#     EVERYTHING = "everything"
#     CHILDREN_ONLY = "children_only"
#     # NOTHING = "nothing"
#
#
# def check_is_arg(arg):
#     return not isinstance(arg, ScopeLevels) and arg is not None
#
# print(check_is_arg(None))
# print(check_is_arg(ScopeLevels.EVERYTHING))
# print(check_is_arg(ScopeLevels.CHILDREN_ONLY))
# print(check_is_arg("everything"))
# print(check_is_arg("children_only"))