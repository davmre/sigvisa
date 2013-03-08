import numpy as np
from django import template

register = template.Library()

@register.filter(name='log')
def log(value):
    return np.log(float(value))

@register.filter(name='div')
def div(value, arg):
    return float(value) / float(arg)
