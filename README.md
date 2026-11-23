Code accompanying the paper 'Remixing Quasi-Normal Modes: Exploring How Detector Calibration Errors Distort
Black Hole Spectroscopy'.

pipeline.py: main functionality, with 1 function per task (e.g. make 1 frame, set up a pyRing run, run 1 set of CVB calculations). Importantly, this includes the implementation of our heuristic calculation (see the 'ansatz' function). 

qnm_fisher_toolkit.py: CVB formula implementation. Called from pipeline.py. (Note: 'teob'/'TEOB' refers to a numerical calculation with the pyRing KerrPostmerger waveform, referred to as Numerical CVB within the paper.)  

pipeline_utils.py: helper functions, called by pipeline.py and fisher_toolkit.py.

All files contain extensive comments and doc strings - see the files themselves for further documentation. 
