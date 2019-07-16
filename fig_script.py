#!/usr/bin/env python
# -*- coding: utf-8 -*-  

import main_module as mm

print 'Epidemiology of culture data analysis v.1.0 Copyright (C) 2019  Richard Walker & Camille Ruiz'
mm.run_experiment("No equatorials.txt")
mm.run_experiment("No equatorials exact direct.txt")
mm.run_experiment("Timmermann no equatorials.txt")
mm.run_experiment("Timmermann no equatorials exact direct.txt")
mm.run_experiment("france_spain.txt")
mm.run_experiment("Australia.txt")
mm.run_experiment("No equatorials gt 10k.txt")
mm.run_experiment("No equatorials lt 10k.txt")