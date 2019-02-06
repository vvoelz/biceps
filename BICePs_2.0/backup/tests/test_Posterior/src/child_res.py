from restraint_cs_H import *    # test (until compute sse) done for work   Yunhui Ge -- 04/2018
from restraint_J import *
from restraint_pf import *
from restraint_cs_Ha import *
from restraint_cs_N import *
from restraint_cs_Ca import *
from restraint_noe import *

class child_res(restraint_cs_H, restraint_J, restraint_cs_Ha, restraint_cs_N, restraint_cs_Ca, restraint_noe, restraint_pf):
	def __init__(self):
		restraint_cs_H.__init__(self)
        	restraint_cs_Ha.__init__(self)
        	restraint_cs_N.__init__(self)
        	restraint_cs_Ca.__init__(self)
        	restraint_J.__init__(self)
        	restraint_noe.__init__(self)
        	restraint_pf.__init__(self)

