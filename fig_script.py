#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import main_module as mm

print 'Epidemiology of culture data analysis v.1.1 Copyright (C) 2019,2020  Richard Walker & Camille Ruiz'

mm.run_experiment("No equatorials.txt")
mm.run_experiment("No equatorials exact direct calibrated.txt")
mm.run_experiment("Timmermann no equatorials.txt")
mm.run_experiment("france_spain.txt")
mm.run_experiment("Australia.txt")
mm.run_experiment("rest of world.txt")
mm.run_experiment("No equatorials lt 10k.txt")
mm.run_experiment("No equatorials gt 10k.txt")
mm.run_experiment("No equatorials infer missing values.txt")
mm.run_experiment("No equatorials phi_1.5.txt")
mm.run_experiment("No equatorials phi_2.5.txt")
mm.run_experiment("No equatorials phi_6.txt")


