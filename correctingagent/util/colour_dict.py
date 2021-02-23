fruit_dict = {'apple': ['apple'],
'banana' : ['banana'],
'blueberry' : ['blueberry'],
'corn' : ['corn'],
'eggplant': ['eggplant'],
'kaki': ['kaki'],
'lemon': ['lemon'],
'mango': ['mango'],
'orange' : ['orange'],
'pear': ['pear']}

fruit_names = []
for v in fruit_dict.values():
    fruit_names.extend(v)

colour_dict = {'red': ['red', 'darkred', 'maroon', 'firebrick', 'crimson'],
'green': ['olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow', 'lawngreen', 'palegreen', 'forestgreen', 'limegreen', 'green', 'seagreen'],
'blue': ['cornflowerblue', 'royalblue', 'midnightblue', 'navy', 'darkblue', 'blue', 'mediumblue', 'dodgerblue', 'deepskyblue'],
'yellow': ['yellow', 'lightyellow', 'lightgoldenrodyellow'],
'purple': ['indigo', 'darkorchid', 'darkviolet', 'purple', 'blueviolet', 'darkmagenta'],
'pink': ['fuchsia', 'hotpink', 'deeppink', 'pink', 'lightpink', 'magenta'],
'orange': ['orange', 'darkorange', 'bisque']}

colour_names = ['charteuse']
for v in colour_dict.values():
    colour_names.extend(v)


simple_colour_dict = {'red': ['red'],
'green': ['green'],
'blue': ['blue'],
'yellow': ['yellow'],
'purple': ['purple'],
'pink': ['pink'],
'orange': ['orange']}

