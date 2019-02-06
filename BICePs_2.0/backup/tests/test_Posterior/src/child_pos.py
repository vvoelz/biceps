from posterior_cs_H import *    # test (until compute sse) done for work   Yunhui Ge -- 04/2018
from posterior_J import *
from posterior_pf import *
from posterior_cs_Ha import *
from posterior_cs_N import *
from posterior_cs_Ca import *
from posterior_noe import *

class child_pos(posterior_cs_H, posterior_J, posterior_cs_Ha, posterior_cs_N, posterior_cs_Ca, posterior_noe, posterior_pf):
	def __init__(self):
		posterior_cs_H.__init__(self)
        	posterior_cs_Ha.__init__(self)
        	posterior_cs_N.__init__(self)
        	posterior_cs_Ca.__init__(self)
        	posterior_J.__init__(self)
        	posterior_noe.__init__(self)
        	posterior_pf.__init__(self)

