from optparse import OptionParser
import os,sys
def input_parser(rt_dir,pot_loc):
    sys.path.insert(0,os.path.join(rt_dir,'lib/'))
    from my_tools import eam_info
    parser = OptionParser()
    parser.add_option("-t", "--temperature",default=200.,help="temperature [K]")
    parser.add_option("-s", "--steps", default=200, help="time on hp")
    parser.add_option("-w", "--wait", default=200, help="wait time before run")
    parser.add_option("-b", "--bins", dest="bins", default=20, help="# hp")
    parser.add_option("-i", "--input", default=None,help="input file", type=str)
    parser.add_option("-m", "--premin", default=0, help="perform pre minimize")
    parser.add_option("-p","--pathway", default='_knots/Fe_DUMB/SC_INFO',help="Path to SC_INFO file")
    parser.add_option("-e","--eam", default='FeP_mm.eam.fs',help="EAM file")
    parser.add_option("-c","--connect", default='cubic',help="spline type")
    (options, args) = parser.parse_args()

    params = {
        'temperature' : float(options.temperature),\
        'bins' : int(options.bins),\
        'steps' : int(options.steps),\
        'therm' : int(options.wait),\
        'potential_file' : os.path.join(pot_loc,options.eam),\
        'pathway' : os.path.join(rt_dir,options.pathway),\
        'expansion' : 1.,\
        'stress' : None,\
        'spline': options.connect,\
        'premin' : bool(options.premin),\
        'overdamped' : True,\
        'com_correction' : True\
    }

    expansion=[]
    if not options.input is None:
        for _line in open(str(options.input)):
            line=_line.strip().split(" ")
            if line[0] == 'pathway':
                params['pathway'] = os.path.join(rt_dir,str(line[1]))
            if line[0] == 'temperature':
                params['temperature'] = float(line[1])
            elif line[0] == 'bins':
                params['bins'] = int(line[1])
            elif line[0] == 'steps':
                params['steps'] = int(line[1])
            elif line[0] == 'therm':
                params['therm'] = int(line[1])
            elif line[0] == 'potential_file':
                params['potential_file'] = os.path.join(pot_loc,str(line[1]))
            elif line[0] == 'expansion':
                for i in range(1,len(line)):
                    expansion.append(float(line[i]))
            elif line[0] == 'stress':
                params['stress'] = [0.,0,0,0.]
                params['stress'][0] = float(line[1])
                params['stress'][1] = int(line[2]) # slip plane
                params['stress'][2] = int(line[3]) # burgers vector index
                params['stress'][3] = float(line[4]) # Area/atom
            elif line[0] == 'spline':
                params['spline'] = str(line[1])
            elif line[0] == 'premin':
                params['premin'] = bool(line[1])
            elif line[0] == 'premin':
                params['pair_style'] = bool(int(line[1]))
            elif line[0] == 'overdamped':
                params['overdamped'] = bool(int(line[1]))
            elif line[0] == 'com_correction':
                params['com_correction'] = bool(line[1])



    if params['spline'] not in ['cubic','quadratic','slinear']:
        params['spline'] == 'slinear'
        print("Invalid scipy spline type, setting to slinear")

    if not 'pair_style' in params:
        params['pair_style'] = 'eam/fs'
        if params['potential_file'].split(".")[1] != 'eam':
            params['pair_style'] = 'meam/spline'

    params['potential'] =  eam_info(params['potential_file'],pair_style=params['pair_style'])
    lattice_constant = params['potential'].lattice
    for c in range(len(expansion)):
        lattice_constant += expansion[c]*(params['temperature']**(c+1))
    params['scale'] = lattice_constant / params['potential'].lattice
    params['mass'] = params['potential'].mass
    params['element'] = params['potential'].ele[0]
    params['cutoff'] = params['potential'].cutoff

    print("""
         _______      _______      _______     _________
        (  ____ )    (  ___  )    (  ____ \    \__   __/
        | (    )|    | (   ) |    | (    \/       ) (
        | (____)|    | (___) |    | (__           | |
        |  _____)    |  ___  |    |  __)          | |
        | (          | (   ) |    | (             | |
        | )          | )   ( |    | )          ___) (___
        |/           |/     \|    |/           \_______/
        Projected    Average      Force        Integrator\n
            (c) TD Swinburne and M-C Marinica 2017
    """)

    print("\n\nINPUT OPTIONS:\n")
    for key,value in params.items():
        print(key,":",value,"\n")
    return params
