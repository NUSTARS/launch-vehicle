#!/usr/bin/env python
import sys
import os 
from datetime import date

# Get today's date
today = date.today()
# today = str(today)
today = today.strftime("%b-%d-%Y")


# Path to template
blank_form_path = "PATH_TO_BLANK_ORDER_FORM"

# Get command line input
x = sys.argv

# Make sure there are enough inputs
if len(x) != 2:
    print('Must enter new form name as input')
    quit()

# Combine the string and create the cmd
name = x[1]
cmd = "cp" + " {}".format(blank_form_path) + " {}".format(today+"_"+name) + ".xlsx"
os.system(cmd)
