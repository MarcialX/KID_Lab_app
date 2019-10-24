"""
Colors for messages. Short cuts to color the messages
"""

COLOR_SEQ = "\033[1;%dm"

HEADER    = '\033[95m'
INFO    = '<font color="blue">'
OK   = '<font color="green">'
WARNING   = '<font color="orange">'
ERROR      = '<font color="red">'
ENDC      = '<\font>'
BOLD      = '\033[1m'
UNDERLINE = '\033[4m'

BLACKTXT, REDTXT, GREENTXT, YELLOWTXT, BLUETXT, MAGENTATXT, CYANTXT, WHITETXT = [30 + i for i in range(8)]
LOGCOLORS = {
    'WARNING' : YELLOWTXT,
    'INFO'    : WHITETXT,
    'DEBUG'   : CYANTXT,
    'CRITICAL': BLUETXT,
    'ERROR'   : REDTXT }