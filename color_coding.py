import re
from termcolor import colored
import streamlit as st
 

def get_color(weight):
    if weight >= 0.02:
        return "green"

    if weight > 0.01 and weight < 0.02:
        return "blue"

    else:
        return "red"    


def get_up_color(topic):
    strngs = ''
    topic_terms = topic[1].split("+")
    for term in topic_terms:
        weight, word = term.split("*")
        weight = float(weight.strip())
        word = word.strip().replace('"', '').replace("'", "")
        color = get_color(weight)
        strng = "<h5 style='color:" + color + "; display: inline-block;'>" +word+"</h5>"
        strng = re.sub(r'\x1b\[\d+m', '',strng)
        strngs = strngs + strng

    return strngs