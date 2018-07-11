colour_dict = {'red': ['red', 'darkred', 'maroon', 'firebrick', 'crimson'],
'green': ['olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow', 'lawngreen', 'palegreen', 'forestgreen', 'limegreen', 'green', 'seagreen'],
'blue': ['cornflowerblue', 'royalblue', 'midnightblue', 'navy', 'darkblue', 'blue', 'mediumblue', 'dodgerblue', 'deepskyblue'],
'yellow': ['yellow', 'lightyellow', 'lightgoldenrodyellow'],
'purple': ['indigo', 'darkorchid', 'darkviolet', 'purple', 'blueviolet', 'darkmagenta'],
'pink': ['fuchsia', 'hotpink', 'deeppink', 'pink', 'lightpink', 'magenta'],
'orange': ['orange', 'darkorange', 'bisque']}

colour_names = []
for v in colour_dict.values():
    colour_names.extend(v)


simple_colour_dict = {'red': ['red'],
'green': ['green'],
'blue': ['blue'],
'yellow': ['yellow'],
'purple': ['purple'],
'pink': ['pink'],
'orange': ['orange']}