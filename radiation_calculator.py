# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:36:20 2016
@author: jakehuneau
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime as dt
from isotopic_dicts import element_dict, iso_amu
import sys

if sys.version_info > (3,0):
    def raw_input(s):
        return input(s)
        
    def xrange(i):
        return range(i)

### Conversions, Constants, Normalizations ###

av_cons = 6.022140858e23

def kev_to_j(keV):
    '''convert keV to Joules'''
    kev_j_conv = 1.60218e-16 #1 keV = 1.60218e-16 J
    return keV * kev_j_conv
    
def decay(z, a, isomer, probability, decay_mode):
    '''
    Uses known information about different decay modes to
    alter z, a, isomer number accordingly to what decay
    mode occurs.
    
    --INPUT--
    z: number of protons (int)
    a: number of nucleons (int)
    isomer: isomer number (int)
    probability: Probability of current isotope (float)
    decay_mode: [decay mode, daughter isomer, branching ratio] (list)
    
    -RETURNS--
    new isotope: (z,a,isomer, probability)
    '''
    if decay_mode[0] == 'beta-':
        z += 1
    elif decay_mode[0] == '2beta-':
        z += 2
    elif decay_mode[0] == 'beta-n':
        z += 1
        a -= 1
    elif decay_mode[0] == 'beta-2n':
        z += 1
        a -= 2
    elif decay_mode[0] == 'beta-3n':
        z += 1
        a -= 3
    elif decay_mode[0] == 'beta-4n':
        z += 1
        a -= 4
    elif decay_mode[0] == 'beta-alpha':
        z += 1
        z -= 2
        a -= 4
    elif decay_mode[0] == 'alpha':
        z -= 2
        a -= 4
    elif decay_mode[0] == 'SF': #spontaneous fission
        print('Spontaneous Fission')
        raise Exception('Decay cannot be found due to Spontaneous fission')
    elif decay_mode[0] == 'IT': #isomeric transformation
        1
    elif decay_mode[0] == 'p':
        z -= 1
        a -= 1
    elif decay_mode[0] == '2p':
        z -= 2
        a -= 2
    elif decay_mode[0] == 'n':
        a -= 1
    elif decay_mode[0] == '2n':
        a -= 2
    elif decay_mode[0] == 'EC/beta+':
        z -= 1
    elif decay_mode[0] == 'EC/beta+p':
        z -= 1
        z -= 1
        a -= 1
    elif decay_mode[0] == 'EC/beta+2p':
        z -= 1
        z -= 2
        a -= 2
    elif decay_mode[0] == 'EC/beta+alpha':
        z -= 1
        z -= 2
        a -= 4
    elif decay_mode[0] == 'EC/beta+SF':
        print('Spontaneous Fission')
        raise Exception('Decay cannot be found due to Spontaneous fission')
    isomer = decay_mode[1]
    probability *= decay_mode[2]
    return (z,a,isomer,probability)
    
def convert_from_activity(activity, species):
    '''
    Converts from activity in mCi to number of atoms. mCi = 3.7x10*13 decays/sec.
    
    --Inputs--
    activity: activity of species in mCi (float)
    decay_const: decay constant in 1/sec (float)
    
    --Returns--
    atom_amnt: number of atoms equivalent to this activity (int)
    '''
    decay_const = np.log(2) / data[species][1]
    decays_per_sec = float(activity) * 3.7e7
    atom_amnt = decays_per_sec / decay_const
    return atom_amnt

def convert_to_activity(atom_amnt, decay_const):
    '''
    Converts number of atoms to milliCurie. Does this using definition of 
    Curie to be 3.7x10^10 decays/sec. 
    https://en.wikipedia.org/wiki/Curie
    
    --Inputs--
    atom_amnt: number of atoms currently (float)
    decay_const: decay constant in 1/sec (float)
    
    --Returns--
    mCi: milliCurie (float)
    '''
    decays_per_sec = atom_amnt * decay_const
    Ci = decays_per_sec / (3.7e10)
    mCi = Ci * 1000
    return mCi
    
def convert_list_to_activity(list_of_atom_amnt, species):
    '''
    Uses convert_to_activity() to convert an entire list of atom amounts to a list
    of activity.
    
    --Inputs--
    list_of_atom_amnt: list of atom amounts, probably from solutions of decay system
    species: (z,a,isomer)
    
    --Returns--
    activity_list: list of solutions for this species but in units mCi
    '''
    activity_list = []
    for atom_amnt in list_of_atom_amnt:
        if data[species][1] != 0:
            species_mCi = convert_to_activity(atom_amnt, np.log(2) / data[species][1])
        else: species_mCi = 0
        
        activity_list.append(species_mCi)
    return activity_list
    
def convert_all_list_to_activity(list_of_list_of_atom_amnt, list_of_species):
    '''
    Uses convert_list_to_activity() to convert an entire list of lists of 
    atom amounts into activity.
    
    --Inputs--
    list_of_list_of_atom_amnt: master list of solutions in unit of number of atoms
    list_of_species: species corresponding to each list of solutions
    
    --Returns--
    master_list_of_activity: list of lists solutions in units mCi
    '''
    master_list_of_activity = []
    for list_of_atom_amnt, species in zip(list_of_list_of_atom_amnt, list_of_species):
        activity_list = convert_list_to_activity(list_of_atom_amnt, species)
        master_list_of_activity.append(activity_list)
    return master_list_of_activity
    
def normalize_dates(dates, time_conv, max_time):
    '''
    Normalize dates so they are in seconds from the first time.
    '''
    normalized_dates = np.array([(dates[i] - dates[0])\
    for i in range(len(dates))])
    normalized_dates = np.array([d.total_seconds() * time_conv for d in \
    normalized_dates])
    
    if not all(normalized_dates[i] <= normalized_dates[i+1] for i in\
     xrange(len(normalized_dates)-1)):
        raise Exception('Dates are not in order.')
            
    if normalized_dates[-1] > max_time:
        raise Exception('Last add time is later than max time.')
            
    return normalized_dates
    
def modify_time(time):
    '''
    Modifies a time in unit seconds into a form that is more
    readable depending on size of time.
    
    --INPUT--
    time: time in seconds (float)
    
    --RETURNS--
    modified_time: time converted to better unit (string)
    '''
    if time == 0.0:
        return 'stable'
    elif time <= 1e-15:
        mod_time = time * 10**18
        return '{} as'.format(mod_time)
    elif 1e-15 < time <= 1e-12:
        mod_time = time * 10**15
        return '{} fs'.format(mod_time)
    elif 1e-12 < time <= 1e-9:
        mod_time = time * 10**12
        return '{} ps'.format(mod_time)
    elif 1e-9 < time <= 1e-6:
        mod_time = time * 10**9
        return '{} ns'.format(mod_time)   
    elif 1e-6 < time <= 1e-3:
        mod_time = time * 10**6
        return '{} us'.format(mod_time)
    elif 1e-3 < time < 1:
        mod_time = time * 10**3
        return '{} ms'.format(mod_time)
    elif 1 <= time <= 60:
        return '{} s'.format(time)
    elif 60 < time <= 6000:
        mod_time = float(time) / 60
        return '{} min'.format(mod_time)
    elif 6000 < time <= 36000:
        mod_time = float(time) / 3600
        return '{} hr'.format(mod_time)
    elif 36000 < time <= 86400000:
        mod_time = float(time) / 86400
        return '{} days'.format(mod_time)
    elif 86400000 < time:
        mod_time = float(time) / 31536000
        return '{} yr'.format(mod_time)

def modify_time_factor(conversion):
    '''
    Used for finding the conversion factor for different times.
    
    --Inputs--
    conversion: string of what conversion is desired. can be the following:
    as attosecond
    fs femtosecond
    ps picosecond
    ns nanosecond
    us microsecond
    ms millisecond
    s second
    min minute
    hr hour
    yr year
    
    --Returns--
    conversion factor
    '''
    if conversion == 'as':
        return 10**18
    elif conversion == 'fs':
        return 10**15
    elif conversion == 'ps':
        return 10**12
    elif conversion == 'ns':
        return 10**9
    elif conversion == 'us':
        return 10**6
    elif conversion == 'ms':
        return 10**3
    elif conversion == 's':
        return 1
    elif conversion == 'min':
        return 1.0/60
    elif conversion == 'hr':
        return 1.0/3600
    elif conversion == 'days' or conversion == 'day':
        return 1.0/86400
    elif conversion == 'yr':
        return 1.0/31536000

### Collect important data and user inputs ###

def create_isotope_data():
    '''
    Uses the database of isotopes from the nndc and creates a
    dictionary from it including relevent information. Format of 
    data = {(z,a,isomer):[isomer_energy, half-life,
    half-life uncertainty, decays]
    '''
    all_lines = []
    data = {}

    f = 'decaydata.txt'
    with open(f,'r') as df:
        next(df)
        next(df)
        for line in df:
            all_lines.append(line.split(';'))

    i = 0
    for i in range(len(all_lines)):
        if all_lines[i][0][0:5] == '     ':
            decay_info = all_lines[i]
            all_lines[i-1].append(decay_info)
        i+=1
    raw_all_info=[]

    for item in all_lines:
        if item[0][0:5] != '     ':
            raw_all_info.append(item)

    for item in raw_all_info:
        z = int(item[0]) #protons
        a = int(item[1]) #nucleons
        isomer = int(item[2]) #isomer number
        is_nrg = float(item[3]) #isomer energy
        t12 = float(item[4]) #half-life
        dt12 = float(item[5]) #half-life uncertainty
        spin = float(item[6]) #nuclear spin
        parity = float(item[7]) #nuclear parity
        ndk = int(item[8]) #number of decay modes
        imp_info = [is_nrg,t12,dt12]
        if len(item) == 10: #means there are decays
            decays = item[9]
            dec_len = len(decays) #checks number of decays
            dm = decays[0].strip() #decay mode
            daughter = int(decays[1]) #daughter isomer
            br = float(decays[2]) #branching ratio
            dbr = float(decays[3]) #branching ratio uncertainty
            Q = float(decays[4]) #Q-value energy
            dQ = float(decays[5].strip()) #Q-value uncertainty
            decay_info = [dm,daughter,br,dbr,Q,dQ]
            imp_info.append(decay_info)
            if dec_len == 7: #2 decay modes
                decay = decays[6]
                dec_len = len(decay)
                dm = decay[0].strip()
                daughter = int(decay[1])
                br = float(decay[2])
                dbr = float(decay[3])
                Q = float(decay[4])
                dQ = float(decay[5].strip())
                decay_info = [dm,daughter,br,dbr,Q,dQ]
                imp_info.append(decay_info)
            if dec_len == 7: #3 decays modes
                decay = decay[6]
                dec_len = len(decay)
                dm = decay[0].strip()
                daughter = int(decay[1])
                br = float(decay[2])
                dbr = float(decay[3])
                Q = float(decay[4])
                dQ = float(decay[5].strip())
                decay_info = [dm,daughter,br,dbr,Q,dQ]
                imp_info.append(decay_info)
            if dec_len == 7: #4 decay modes
                decay = decay[6]
                dec_len = len(decay)
                dm = decay[0].strip()
                daughter = int(decay[1])
                br = float(decay[2])
                dbr = float(decay[3])
                Q = float(decay[4])
                dQ = float(decay[5].strip())
                decay_info = [dm,daughter,br,dbr,Q,dQ]
                imp_info.append(decay_info)
            if dec_len == 7: #5 decay modes
                decay = decay[6]
                dec_len = len(decay)
                dm = decay[0].strip()
                daughter = int(decay[1])
                br = float(decay[2])
                dbr = float(decay[3])
                Q = float(decay[4])
                dQ = float(decay[5].strip())
                decay_info = [dm,daughter,br,dbr,Q,dQ]
                imp_info.append(decay_info)
        data.update({(z,a,isomer):imp_info})
    return data
  
def file_input():
    '''
    Used if data inputted it through a text file
    '''
    multiple_dates = False
    
    file = raw_input('File name: ')
    f = np.loadtxt(file, dtype = 'str')
    
    modify_t12 = raw_input(\
    'What time scale? (as, fs, ps, ns, us, ms, s, min, hr, days, yr): ')
    while modify_t12 not in {'as','fs','ps','ns','us','ms','s','min','hr','days','yr'}:
        print('{} is not a proper time unit'.format(modify_t12))
        modify_t12 = raw_input(\
        'What time scale? (as, fs, ps, ns, us, ms, s, min, hr, days, yr): ')
        
    time_conv = modify_time_factor(modify_t12)
    z,a,isomer,probability = find_iso_in_file_str(file)
    isotope = (z,a,isomer, probability)
    max_time = float(raw_input('Max time ({}): '.format(modify_t12)))
        
    if np.size(f) == 2:
        date = extract_dates(f[0])
        normalized_dates = np.array([0])
        initial_activity = float(f[1])
        initial_mass = convert_from_activity(initial_activity, (z,a,isomer))
        multiple_dates = False
        
    else:
        dates = extract_dates(f[:,0], single=False)
        normalized_dates = normalize_dates(dates, time_conv, max_time)
        added_masses = f[:,1]
        if added_masses[0][0] == 'b':
            activities = [float(added_mass[2:-1]) for added_mass in added_masses]
        else:
            activities = [float(added_mass) for added_mass in added_masses]
            
        masses = np.array([convert_from_activity(act,(z,a,isomer)) \
        for act in activities])
        initial_mass = masses[0]
        multiple_dates = True
        
    return modify_t12, isotope, max_time, normalized_dates, initial_mass, multiple_dates,\
    masses

def manual_input():
    '''
    If data is to be inputed manually
    '''
    multiple_dates = False
    
    z = raw_input('z: ')
    try:
        z = int(z)
    except:
        raise Exception('{} is not a valid atomic number'.format(z))
        
    a = raw_input('a: ')
    try:
        a = int(a)
    except:
        raise Exception('{} is not a valid nucleon number'.format(a))
        
    isomer = raw_input('isomer: ')
    try:
        isomer = int(isomer)
    except:
        raise Exception('{} is not a valid isomer number'.format(isomer))
        
    probability = 1
    initial_activity = raw_input('Initial activity (mCi): ')
    try:
        initial_activity = float(initial_activity)
    except:
        raise Exception('{} is not a valid activity'.format(initial_activity))
        
    initial_mass = convert_from_activity(initial_activity, (z,a,isomer))
    isotope = (z,a,isomer, probability)
    
    modify_t12 = raw_input(\
    'What time scale? (as, fs, ps, ns, us, ms, s, min, hr, days, yr): ')
    while modify_t12 not in {'as','fs','ps','ns','us','ms','s','min','hr','days','yr'}:
        print('{} is not a proper time unit'.format(modify_t12))
        modify_t12 = raw_input(\
        'What time scale? (as, fs, ps, ns, us, ms, s, min, hr, days, yr): ')
        
    max_time = raw_input('Max time: ')
    try:
        max_time = float(max_time)
    except:
        raise Exception('{} is not a valid time'.format(max_time))
        
    normalized_dates = [0]
    masses = [initial_mass]
    
    if raw_input('Any added mass? (yes, no): ').lower() in {'yes','y','ye'}:
        multiple_dates = True
        print("Type 'n' to stop adding")
        while True:
            try:
                addition_time = float(raw_input('Addition time ({}): '.format(\
                modify_t12)))
                addition_activity = float(raw_input('Addition activity (mCi): '))
                addition_amount = convert_from_activity(addition_activity,(z,a,isomer))
                normalized_dates.append(addition_time)
                masses.append(addition_amount)
                    
                if not all(normalized_dates[i] <= normalized_dates[i+1] for i in\
                 xrange(len(normalized_dates)-1)):
                    raise Exception('Dates are not in order.')           
            except:
                break
                
    return modify_t12, isotope, max_time, normalized_dates, initial_mass, multiple_dates,\
     masses

def data_input():
    '''
    Asks for type of input for program.
    '''
    input_type = raw_input('file or manual? ').lower()
    
    if input_type == 'file' or input_type == 'f':
        modify_t12, isotope, max_time, normalized_dates, initial_mass, multiple_dates,\
        masses = file_input()
        
    elif input_type == 'manual' or input_type == 'm':
        modify_t12, isotope, max_time, normalized_dates, initial_mass, multiple_dates,\
        masses = manual_input()
        
    else:
        raise Exception("You didn't input file or manual.")
    
    plot_type = raw_input('log plot or linear? ').lower()
    if plot_type != 'log' and plot_type != 'linear' and plot_type != 'lin':
        raise Exception('{} is not an allowed plot type'.format(plot_type))
    
    return modify_t12, isotope, max_time, normalized_dates, initial_mass, multiple_dates,\
        masses, plot_type
        
def find_iso_in_file_str(file_str):
    '''
    Finds the isomer details from the filename. File name must be in format of
    z.a.isomer.~~~.txt where ~~~ can be anything.
    
    --Input--
    file_str: file name
    
    --Returns--
    isotope: (z,a,isomer, probability) where probability = 1 because it is initial
    '''
    first_stop = file_str.find('.')
    file_str = file_str.replace('.','-',1)
    second_stop = file_str.find('.')
    file_str = file_str.replace('.','-',1)
    third_stop = file_str.find('.')
    z = int(file_str[:first_stop])
    a = int(file_str[first_stop+1:second_stop])
    isomer = int(file_str[second_stop+1:third_stop])
    return (z,a,isomer,1)
    
def make_string_iso_into_tuple(string_iso):
    '''
    Makes a string isotope into a tuple isotope format.
    
    --Inputs--
    string_iso: string in format '(z,a,isomer)'
    
    --Returns--
    (z,a,iso)
    '''
    numbers = string_iso[1:-1]
    z = int(numbers[:numbers.index(',')])
    numbers = numbers[numbers.index(',')+1:]
    a = int(numbers[:numbers.index(',')])
    numbers = numbers[numbers.index(',')+1:]
    iso = int(numbers)
    return (z,a,iso)
    
def extract_dates(dates, single = True):
    '''
    Extracts dates to a workable format from a the following datetime format:
    year.month.day(.hour(.minute(.second))) where anything in () is optional.
    '''
    if dates[0][0] == 'b':
        dates = [date[2:-1] for date in dates]
        
    if single:
        if dates.count('.') == 2:
            date = dt.datetime.strptime(f[0],'%Y.%m.%d').date()
        elif dates.count('.') == 3:
            date = dt.datetime.strptime(f[0],'%Y.%m.%d.%H')
        elif dates.count('.') == 4:
            date = dt.datetime.strptime(f[0],'%Y.%m.%d.%H.%M')
        elif dates.count('.') == 5:
            date = dt.datetime.strptime(f[0],'%Y.%m.%d.%H.%M.%S')
        return date
    else:
        if dates[0].count('.') == 2:
            dates = [dt.datetime.strptime(d,'%Y.%m.%d').date() for d in dates]
        elif dates[0].count('.') == 3:
            dates = [dt.datetime.strptime(d,'%Y.%m.%d.%H') for d in dates] 
        elif dates[0].count('.') == 4:
             dates = [dt.datetime.strptime(d,'%Y.%m.%d.%H.%M') for d in dates]
        elif dates[0].count('.') == 5:
            dates = [dt.datetime.strptime(d,'%Y.%m.%d.%H.%M.%S') for d in dates]
        return dates
               
### printing, saving, displaying ###

def print_exp_dict(exposure_dict):
    '''
    Used for printing out only non-zero radiation energies.
    '''
    print('\n Radiation Energies')
    for rad in exposure_dict:
        if exposure_dict[rad] != 0:
            print(rad, ':', exposure_dict[rad], 'J')

def create_plots(plot_type, solns, spec_names):
    f, axarr = plt.subplots(2,1, figsize = (12,8), sharex=True)
    
    if plot_type == 'lin' or plot_type == 'linear':  
    
        for soln, name in zip(solns, spec_names):
            axarr[0].plot(time, soln, label=name)
        for soln, name in zip(activity_solns, spec_names):
            axarr[1].plot(time, soln, label=name)
            
    elif plot_type == 'log':
        for soln, name in zip(solns, spec_names):
            axarr[0].loglog(time, soln, label=name)
        for soln, name in zip(activity_solns, spec_names):
            axarr[1].loglog(time, soln, label=name)
    
    box = axarr[0].get_position()
    axarr[0].set_position([box.x0, box.y0, box.width * 0.9, box.height])     

    box = axarr[1].get_position()
    axarr[1].set_position([box.x0, box.y0, box.width * 0.9, box.height])   
       
    axarr[0].legend(loc='center left', bbox_to_anchor=(1, 0))
    axarr[0].set_title('decay of ({},{},{})'.format(z,a,isomer))
    axarr[1].set_xlabel('time ({})	'.format(modify_t12))
    axarr[0].set_ylabel('Number of atoms')
    axarr[1].set_ylabel('Activity (mCi)')
    plt.show()
    
def save_solns(z,a,isomer, modify_t12, spec_names, solns, time):
    save_soln = np.transpose(np.insert(solns, 0, time, axis=0))
    
    np.savetxt("{}.{}.{}.out.txt".format(z,a,isomer), save_soln, \
    header = 'time({}) '.format(modify_t12)+' '.join(spec_names))

### Analyze Isotope ### 

def check_stable(isotope):
    '''
    Checks stability of isotope. Checks if there are any decay modes.
    '''
    z,a,isomer, probability = isotope
    try: isotope_info = data[(z,a,isomer)]
    except KeyError: raise Exception('This isotope does not exist')
    if len(isotope_info) > 3: return False #decays
    else: return True

def find_daughters(isotope):
    '''
    Creates list of daughter isotopes. Does this according to
    to each decay mode of parent isotope.
    
    --INPUT--
    isotope: parent isotope in form (z,a,isomer, probability) (tuple)
    
    --RETURNS--
    daughters: list of 5 possible daugther isotopes. list will
    be of length 5, and if there are only n number of daughter
    isotopes, then first n items of list will be daughters
    and all other items in list will be None.
    '''
    z,a,isomer, probability = isotope
    isotope_info = data[(z,a,isomer)]
    if len(isotope_info) <= 3: raise('This is stable')
    else:
        decays = isotope_info[3:]
        decay_modes = [item[:3] for item in decays]
        daughters = [decay(z,a,isomer, probability, mode)\
         for mode in decay_modes]
    while len(daughters) < 5:
        daughters += [None]
    return daughters
    
def find_parents(init_strings, ordered_terms):
    '''
    Used for finding the parents for every species in the chain
    '''
    parent_dict = {}
    for spec,equation in zip(init_strings,ordered_terms):
        parents = []
        for term in equation:
            if term[0] > 0: #not stable
                parents.append(term[1][:-1])
        parent_dict.update({spec[:-1]:parents})
    return parent_dict
    
### Build decay chains ###

class Node(object):
    def __init__(self,isotope,parent):
        self._parent = parent
        self._isotope = isotope
        self._satisfied = isotope == None or check_stable(isotope)
        self._branch1 = None
        self._branch2 = None
        self._branch3 = None
        self._branch4 = None
        self._branch5 = None
        self._depth = parent.depth + 1 if parent != None else 1

    @property
    def parent(self):
        return self._parent

    @property
    def isotope(self):
        return self._isotope

    @property
    def satisfied(self):
        return self._satisfied

    @property
    def depth(self):
        return self._depth

    @property
    def branch1(self):
        return self._branch1

    @branch1.setter
    def branch1(self,value):
        self._branch1 = value

    @property
    def branch2(self):
        return self._branch2

    @branch2.setter
    def branch2(self,value):
        self._branch2 = value
        
    @property
    def branch3(self):
        return self._branch3

    @branch3.setter
    def branch3(self,value):
        self._branch3 = value
        
    @property
    def branch4(self):
        return self._branch4

    @branch4.setter
    def branch4(self,value):
        self._branch4 = value
        
    @property
    def branch5(self):
        return self._branch5

    @branch5.setter
    def branch5(self,value):
        self._branch5 = value
        
def create_all_chains(node,chain=[]):
    '''
    Recursive function for building tree full of decays. Not totally
    sure how this works.
    '''
    if node.branch1 is None:
        chain.append(node.isotope)
        if chain[-1] != None:
            master.append(chain)
    else:
        create_all_chains(node.branch1, chain[:] + [node.isotope])
        create_all_chains(node.branch2, chain[:] + [node.isotope])
        create_all_chains(node.branch3, chain[:] + [node.isotope])
        create_all_chains(node.branch4, chain[:] + [node.isotope])
        create_all_chains(node.branch5, chain[:] + [node.isotope])
    return master

def build_tree(node, maxDepth):
    '''
    Recursive function for building down tree until isotope is 
    stable. maxDepth is how far down tree can go until it stops
    trying to go further.
    '''
    if not node.satisfied and node.depth<maxDepth:
        daughters = find_daughters(node.isotope)
        node.branch1 = Node(daughters[0], node)
        build_tree(node.branch1,maxDepth)
        
        node.branch2 = Node(daughters[1], node)
        build_tree(node.branch2,maxDepth)
        
        node.branch3 = Node(daughters[2], node)
        build_tree(node.branch3,maxDepth)
        
        node.branch4 = Node(daughters[3], node)
        build_tree(node.branch4,maxDepth)
        
        node.branch5 = Node(daughters[4], node)
        build_tree(node.branch5,maxDepth)
        
def find_decay(isotope):
    '''
    Builds entire tree of decay. master must be an empty list
    made before function is called, which will be filled by each
    decay chain.
    '''
    z, a, isomer, probability = isotope
    root = Node(isotope,None)
    build_tree(root,maxDepth=25)
    master = create_all_chains(root)

def add_timecons(master_list, app_t12 = False, modify_t12 = False):
    '''
    Adds time constant to master list of all decay chains.
    half-life = log(2)/time_const
    
    --INPUT--
    master_list: list of all decay chains (list)
    t12: True = half-life will be calculated instead of time constants
    modify_t12: True = half-lives will be modifed into strings with
    nice unit, False = half-lives in seconds, also can be
    femtoseconds, nanoseconds,milliseconds, seconds, days, years
    (True/False/'fs'/'ns'/'ms'/'s'/'days'/'years')
    
    --RETURNS--
    new_master_list: list of decay chains with half lives (list)
    '''
    new_master_list = []
    for chain in master_list:
        new_chain = []
        for isotope in chain:
            z = isotope[0]
            a = isotope[1]
            isomer = isotope[2]
            isotope_info = data[(z,a,isomer)]
            t12 = isotope_info[1]
            if modify_t12 == 'as': t12 *= 10**18
            elif modify_t12 == 'fs': t12 *= 10**15
            elif modify_t12 == 'ps': t12 *= 10**12
            elif modify_t12 == 'ns': t12 *= 12**9
            elif modify_t12 == 'ms': t12 *= 10**3
            elif modify_t12 == 's': t12 = t12
            elif modify_t12 == 'min': t12 /= 60
            elif modify_t12 == 'hr': t12 /= 3600
            elif modify_t12 == 'days': t12 /= 86400
            elif modify_t12 == 'yr': t12 /= 31536000
            elif modify_t12: t12 = modify_time(t12)
            time_const = t12 / np.log(2)
            if app_t12:
                isotope = isotope + (t12,)
            else:
                isotope = isotope + (time_const,)
            new_chain.append(isotope)
        new_master_list.append(new_chain)
    return new_master_list

### Decay Equations ###
  
def equation_builder(master_chain):
    '''
    Sorts through all the decay chains and builds the equations to be solved
    later on. Description for how equations are made is found at 
    https://github.com/TheStrangeQuark/Radiation_Calculator/blob/
    master/Decay_Chain_Equations.pdf
    Works by checking each decay and using an equation for creating the equation
    for the amount of the species remaining.
    
    --Input--
    master_chain: list of all decay chains
    
    --Returns--
    master_equations: dictionary of all equations for amount of each isotope.
    '''
    master_equations = {}
    m = 0 #keep track of each master chain so species are not doubled
    for chain in master_chain:
        n = 0 #keep track of species in an individual chain
        for item in chain:
            if item == chain[0]:
                equation = {'({},{},{})f'.format(item[0],item[1],item[2]):\
                '-({},{},{})i/{}'.format(item[0],item[1],item[2],item[4])}
                master_equations.update(equation)
                
            elif '({},{},{})f'.format(item[0],item[1],item[2]) not in master_equations:
                equation = {'({},{},{})f'.format(item[0],item[1],item[2]) :\
                str(item[3]/chain[n-1][3])+'('+str(chain[n-1][0])+','+\
                str(chain[n-1][1])+','+str(chain[n-1][2])+')i/'+str(chain[n-1][4])+\
                '-('+str(item[0])+','+str(item[1])+','+str(item[2])+\
                ')i/'+str(item[4])}
                master_equations.update(equation)
            else:
                try:
                    if '({},{},{})f'.format(item[0],item[1],item[2]) in master_equations\
                    and master_chain[m-1][n-1] != master_chain[m][n-1]:
                        old_equation = master_equations['({},{},{})f'.format\
                        (item[0],item[1],item[2])]
                        new_equation = old_equation + '+' +\
                        str(item[3]/chain[n-1][3])+'('+str(chain[n-1][0]) +','+\
                        str(chain[n-1][1])+','+str(chain[n-1][2])+')i/'+str(chain[n-1][4])
                        equation = {'({},{},{})f'.format(item[0],item[1],item[2]) :\
                         new_equation}
                        master_equations.update(equation)
                except IndexError:
                    if '({},{},{})f'.format(item[0],item[1],item[2]) in master_equations\
                    and master_chain[m-1][-1] != master_chain[m][n-1]:
                        old_equation = master_equations['({},{},{})f'.format\
                        (item[0],item[1],item[2])]
                        new_equation = old_equation + '+' +\
                        str(item[3]/chain[n-1][3])+'('+str(chain[n-1][0]) +','+\
                        str(chain[n-1][1])+','+str(chain[n-1][2])+')i/'+str(chain[n-1][4])
                        equation = {'({},{},{})f'.format(item[0],item[1],item[2]) :\
                         new_equation}
                        master_equations.update(equation)
            n += 1
        m += 1         
    return master_equations
    
def find_modifier(string):
    '''
    Finds the first instance of either + or - in a string while ignoring
    e+ or e- that would be part of a number.
    
    --Input--
    string
    
    --Returns--
    location of first instance of either + or - or length of string if neither
    are present.
    '''
    string_len = len(string)
    if 'e+' in string : string = string.replace('e+', 'ee')
    
    if '+' in string: plus_loc = string.find('+')
    else: plus_loc = string_len
    
    if 'e-' in string: string = string.replace('e-','ee')
    if '-' in string: min_loc = string.find('-')
    else: min_loc = string_len
    
    if plus_loc < min_loc: return plus_loc
    elif min_loc < plus_loc: return min_loc
    else: return string_len+1

def process_equation(equation):
    '''
    Processes an equation made in equation_builder() to break into
    terms of equation, which is a more usable format. Works by finding 
    species in each term of the equation and seperating it from the 
    probability of decay and half-life associated with the species. If 
    the isotope is stable, then the term is 0 because half-life is 
    infinite, so 1/inf = 0.
    
    --Input--
    equation: string equation that describes the amount of mass of a species
    
    --Returns--
    terms: list with terms involved in equation. Format is [[factor, species]]
    where the factor is what the species is being multiplied by. List may also be
    [[factor, species1], [factor, species2],...[factor,species_n]] for n species.
    '''
    terms = []
    while True:
        if len(equation) == 0:
            break
            
        if len(equation) > 0 and equation[0] == ' ': #skip over any spaces
            equation = equation[1:]
            
        if len(equation) > 0 and equation[0] in '0123456789': #starts with factor
            species_start = equation.find('(')
            species_end = equation.find('i')+1
            species = equation[species_start:species_end]
            branch = equation[:species_start]
            mod_loc = find_modifier(equation)
            half_life = equation[species_end+1:mod_loc]
            if float(half_life) != 0.0:
                terms.append([float(branch)/float(half_life), species])
            elif float(half_life) == 0.0:
                    terms.append([0, species])
            equation = equation[mod_loc:]
        
        if len(equation) > 0 and equation[0] == '(': #starts with species
            species_end = equation.find('i')+1
            mod_loc = find_modifier(equation)
            half_life = equation[species_end+1:mod_loc]
            if float(half_life) != 0.0:
                terms.append([1.0/float(half_life), species])
            elif float(half_life) == 0.0:
                    terms.append([0, species])
            equation = equation[mod_loc:]
        
        if len(equation) > 0 and equation[0] == '-': #subtracting factors
            equation = equation[1:] #skip over subtract and make factor neg
            if equation[0] in '0123456789':
                species_start = equation.find('(')
                species_end = equation.find('i')+1
                species = equation[species_start:species_end]
                branch = equation[:species_start]
                species = equation[species_start:species_end]
                mod_loc = find_modifier(equation)
                half_life = equation[species_end+1:mod_loc]
                if float(half_life) != 0.0:
                    terms.append([-1*float(branch)/float(half_life), species])
                elif float(half_life) == 0.0:
                    terms.append([0, species])
                equation = equation[mod_loc:]
                
            elif equation[0] == '(':
                species_end = equation.find('i')+1
                species = equation[:species_end]
                mod_loc = find_modifier(equation)
                half_life = equation[species_end+1:mod_loc]
                if float(half_life) != 0.0:
                    terms.append([-1.0/float(half_life), species])
                elif float(half_life) == 0.0:
                    terms.append([0, species])
            equation = equation[mod_loc:]
        
        if len(equation) > 0 and equation[0] == '+': #adding factors
            equation = equation[1:]
            if equation[0] in '0123456789':
                species_start = equation.find('(')
                species_end = equation.find('i')+1
                species = equation[species_start:species_end]
                branch = equation[:species_start]
                mod_loc = find_modifier(equation)
                half_life = equation[species_end+1:mod_loc]
                if float(half_life) != 0.0:
                    terms.append([1*float(branch)/float(half_life), species])
                elif float(half_life) == 0.0:
                    terms.append([0, species])
                    
            elif equation[0] == '(':
                species_end = equation.find('i')+1
                equation = equation[1:]
                mod_loc = find_modifier(equation)
                half_life = equation[species_end+1:mod_loc]
                if float(half_life) != 0.0:
                    terms.append([1.0/float(half_life), species])
                elif float(half_life) == 0.0:
                    terms.append([0, species])
                equation = equation[mod_loc:]
        break
    return terms

### process decay ###

def create_init_dict(init_species, init_species_amnt, master_terms):
    '''
    Creates dictionary that will make initial values for each isotope.
    Assumes there is only an initial amount of the first species in the decay
    chain. Checks if each initial species in each term is the initial species
    and if not, then sets equal to 0.
    
    --Inputs--
    init_species: tuple in form (z,a,isomer)
    init_species_amnt: float of the amount of the initial species
    master_terms: dictionary of all the equations
    
    --Returns--
    init_dict: dictionary of all initial amounts for each species in the form
    {'species' : amount}
    '''
    init_dict = {}
    z,a,isomer = init_species
    init_species_str = '({},{},{})i'.format(z,a,isomer)
    init_dict.update({init_species_str : init_species_amnt})
    for equation in master_terms:
        for term in master_terms[equation]:
            if term[1] != init_species_str: init_dict.update({term[1]:0})
    return init_dict
    
def create_decay_order(master_decay_list):
    '''
    Creates order of species in decay tree. parallel species are ordered left
    to right.
    
    --Input--
    master_decay_list: list of all decay chains created using find_decay()
    
    --Returns--
    all_species: list of species in order of initial isotope to last
    '''
    n = 0 #keep track of which item in decay chains
    all_species = []
    longest_chain = 0
    for chain in master_decay_list:
        if len(chain) > longest_chain: longest_chain = len(chain)
        
    while n < longest_chain:
        for decay_chain in master_decay_list: #look at each decay chain
            try:
                if decay_chain[n][0:3] not in all_species:
                    all_species.append(decay_chain[n][0:3])
            except:
                continue
        n+=1
    return all_species

def remove_duplicates_nest_list(a_list):
    '''
    Removes duplicates from a nested list in form [[],[],...,[]] by 
    transforming each nested list into a tuple and making a set from all
    the tuples then making that set back into a list of lists.
    
    --Inputs--
    a_list: nested list
    
    --Returns--
    same list but without duplicates
    '''
    return [list(tup) for tup in set([tuple(x) for x in a_list])]

def create_ordered_terms(init_strings, master_terms):
    '''
    uses ordered list of isotopes to create list of terms in proper order
    
    --Inputs--
    init_strings: list of isotopes in proper order
    master_terms: dictionary of all terms made from equations
    
    --Returns--
    ordered_terms: list of all ordered terms in form 
    [[coeff, 'species1'],..[coeff, 'speciesn']
    '''
    ordered_terms = []
    for iso in init_strings:
        iso = iso.replace('i','f')
        terms = remove_duplicates_nest_list(master_terms[iso])
        ordered_terms.append(terms)
    return ordered_terms

### Calculate solutions from equations ###

def decaychain(init, t):
    '''
    Systems of equations for different possible decays. Needs to check for 
    length of decay because python cannot dynamically create fn for n number
    of decays. Used for solving in initial_chain() which will calculate 
    amount of each isotope.
    
    --Inputs--
    init: list of amount of each isotope. usually [n,0,0,0,...,j] for a decay
    with j decays and n initial amount of the first isotope.
    t: list of times 
    
    --Returns--
    [f0,f1,...,fn]: new amount of each isotope
    '''
    if len(init) == 1:
        return [init[0] for time in t]
    elif len(init) == 2:
        iso0, iso1 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        return [f0, f1]
    elif len(init) == 3:
        iso0, iso1, iso2 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        return [f0,f1,f2]
    elif len(init) == 4:
        iso0, iso1, iso2, iso3 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        return [f0,f1,f2,f3]
    elif len(init) == 5:
        iso0, iso1, iso2, iso3, iso4 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        return [f0,f1,f2,f3,f4]
    elif len(init) == 6:
        iso0, iso1, iso2, iso3, iso4, iso5 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        return [f0,f1,f2,f3,f4,f5]
    elif len(init) == 7:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        return [f0,f1,f2,f3,f4,f5,f6]
    elif len(init) == 8:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6,iso7 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        return [f0,f1,f2,f3,f4,f5,f6, f7]
    elif len(init) == 9:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8]
    elif len(init) == 10:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9]
    elif len(init) == 11:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10]
    elif len(init) == 12:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5,init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11]
    elif len(init) == 13:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12]
    elif len(init) == 14:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13]
    elif len(init) == 15:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]
    elif len(init) == 16:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15]
    elif len(init) == 17:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]
    elif len(init) == 18:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16, iso17 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16,\
        init_strings[17]:iso17}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        f17 = np.sum([ordered_terms[17][n][0]*init_dict_str[ordered_terms[17][n][1]]\
         for n in range(len(ordered_terms[17]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17]
    elif len(init) == 19:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16, iso17, iso18 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16,\
        init_strings[17]:iso17, init_strings[18]:iso18}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        f17 = np.sum([ordered_terms[17][n][0]*init_dict_str[ordered_terms[17][n][1]]\
         for n in range(len(ordered_terms[17]))])
        f18 = np.sum([ordered_terms[18][n][0]*init_dict_str[ordered_terms[18][n][1]]\
         for n in range(len(ordered_terms[18]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18]
    elif len(init) == 20:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16, iso17, iso18, iso19 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16,\
        init_strings[17]:iso17, init_strings[18]:iso18, init_strings[19]:iso19}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        f17 = np.sum([ordered_terms[17][n][0]*init_dict_str[ordered_terms[17][n][1]]\
         for n in range(len(ordered_terms[17]))])
        f18 = np.sum([ordered_terms[18][n][0]*init_dict_str[ordered_terms[18][n][1]]\
         for n in range(len(ordered_terms[18]))])
        f19 = np.sum([ordered_terms[19][n][0]*init_dict_str[ordered_terms[19][n][1]]\
         for n in range(len(ordered_terms[19]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,\
        f18,f19]
    elif len(init) == 21:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16, iso17, iso18, iso19,\
        iso20 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16,\
        init_strings[17]:iso17, init_strings[18]:iso18, init_strings[19]:iso19,\
        init_strings[20]:iso20}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        f17 = np.sum([ordered_terms[17][n][0]*init_dict_str[ordered_terms[17][n][1]]\
         for n in range(len(ordered_terms[17]))])
        f18 = np.sum([ordered_terms[18][n][0]*init_dict_str[ordered_terms[18][n][1]]\
         for n in range(len(ordered_terms[18]))])
        f19 = np.sum([ordered_terms[19][n][0]*init_dict_str[ordered_terms[19][n][1]]\
         for n in range(len(ordered_terms[19]))])
        f20 = np.sum([ordered_terms[20][n][0]*init_dict_str[ordered_terms[20][n][1]]\
         for n in range(len(ordered_terms[20]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,\
        f18,f19,f20]
    elif len(init) == 22:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16, iso17, iso18, iso19,\
        iso20, iso21 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16,\
        init_strings[17]:iso17, init_strings[18]:iso18, init_strings[19]:iso19,\
        init_strings[20]:iso20, init_strings[21]:iso21}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        f17 = np.sum([ordered_terms[17][n][0]*init_dict_str[ordered_terms[17][n][1]]\
         for n in range(len(ordered_terms[17]))])
        f18 = np.sum([ordered_terms[18][n][0]*init_dict_str[ordered_terms[18][n][1]]\
         for n in range(len(ordered_terms[18]))])
        f19 = np.sum([ordered_terms[19][n][0]*init_dict_str[ordered_terms[19][n][1]]\
         for n in range(len(ordered_terms[19]))])
        f20 = np.sum([ordered_terms[20][n][0]*init_dict_str[ordered_terms[20][n][1]]\
         for n in range(len(ordered_terms[20]))])
        f21 = np.sum([ordered_terms[21][n][0]*init_dict_str[ordered_terms[21][n][1]]\
         for n in range(len(ordered_terms[21]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,\
        f18,f19,f20,f21]
    elif len(init) == 23:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16, iso17, iso18, iso19,\
        iso20, iso21, iso22 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16,\
        init_strings[17]:iso17, init_strings[18]:iso18, init_strings[19]:iso19,\
        init_strings[20]:iso20, init_strings[21]:iso21, init_strings[22]:iso22}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        f17 = np.sum([ordered_terms[17][n][0]*init_dict_str[ordered_terms[17][n][1]]\
         for n in range(len(ordered_terms[17]))])
        f18 = np.sum([ordered_terms[18][n][0]*init_dict_str[ordered_terms[18][n][1]]\
         for n in range(len(ordered_terms[18]))])
        f19 = np.sum([ordered_terms[19][n][0]*init_dict_str[ordered_terms[19][n][1]]\
         for n in range(len(ordered_terms[19]))])
        f20 = np.sum([ordered_terms[20][n][0]*init_dict_str[ordered_terms[20][n][1]]\
         for n in range(len(ordered_terms[20]))])
        f21 = np.sum([ordered_terms[21][n][0]*init_dict_str[ordered_terms[21][n][1]]\
         for n in range(len(ordered_terms[21]))])
        f22 = np.sum([ordered_terms[22][n][0]*init_dict_str[ordered_terms[22][n][1]]\
         for n in range(len(ordered_terms[22]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,\
        f18,f19,f20,f21,f22]
    elif len(init) == 24:
        iso0, iso1, iso2, iso3, iso4, iso5, iso6, iso7, iso8, iso9, iso10,\
        iso11, iso12, iso13, iso14, iso15, iso16, iso17, iso18, iso19,\
        iso20, iso21, iso22, iso23 = init
        init_dict_str = {init_strings[0]:iso0,init_strings[1]:iso1,\
        init_strings[2]:iso2, init_strings[3]:iso3, init_strings[4]:iso4,\
        init_strings[5]:iso5, init_strings[6]:iso6, init_strings[7]:iso7,\
        init_strings[8]:iso8, init_strings[9]:iso9, init_strings[10]:iso10,\
        init_strings[11]:iso11, init_strings[12]:iso12, init_strings[13]:iso13,\
        init_strings[14]:iso14, init_strings[15]:iso15, init_strings[16]:iso16,\
        init_strings[17]:iso17, init_strings[18]:iso18, init_strings[19]:iso19,\
        init_strings[20]:iso20, init_strings[21]:iso21, init_strings[22]:iso22,\
        init_strings[23]:iso23}
        f0 = ordered_terms[0][0][0] * init_dict_str[ordered_terms[0][0][1]]
        f1 = np.sum([ordered_terms[1][n][0]*init_dict_str[ordered_terms[1][n][1]]\
         for n in range(len(ordered_terms[1]))])
        f2 = np.sum([ordered_terms[2][n][0]*init_dict_str[ordered_terms[2][n][1]]\
         for n in range(len(ordered_terms[2]))])
        f3 = np.sum([ordered_terms[3][n][0]*init_dict_str[ordered_terms[3][n][1]]\
         for n in range(len(ordered_terms[3]))])
        f4 = np.sum([ordered_terms[4][n][0]*init_dict_str[ordered_terms[4][n][1]]\
         for n in range(len(ordered_terms[4]))])
        f5 = np.sum([ordered_terms[5][n][0]*init_dict_str[ordered_terms[5][n][1]]\
         for n in range(len(ordered_terms[5]))])
        f6 = np.sum([ordered_terms[6][n][0]*init_dict_str[ordered_terms[6][n][1]]\
         for n in range(len(ordered_terms[6]))])
        f7 = np.sum([ordered_terms[7][n][0]*init_dict_str[ordered_terms[7][n][1]]\
         for n in range(len(ordered_terms[7]))])
        f8 = np.sum([ordered_terms[8][n][0]*init_dict_str[ordered_terms[8][n][1]]\
         for n in range(len(ordered_terms[8]))])
        f9 = np.sum([ordered_terms[9][n][0]*init_dict_str[ordered_terms[9][n][1]]\
         for n in range(len(ordered_terms[9]))])
        f10 = np.sum([ordered_terms[10][n][0]*init_dict_str[ordered_terms[10][n][1]]\
         for n in range(len(ordered_terms[10]))])
        f11 = np.sum([ordered_terms[11][n][0]*init_dict_str[ordered_terms[11][n][1]]\
         for n in range(len(ordered_terms[11]))])
        f12 = np.sum([ordered_terms[12][n][0]*init_dict_str[ordered_terms[12][n][1]]\
         for n in range(len(ordered_terms[12]))])
        f13 = np.sum([ordered_terms[13][n][0]*init_dict_str[ordered_terms[13][n][1]]\
         for n in range(len(ordered_terms[13]))])
        f14 = np.sum([ordered_terms[14][n][0]*init_dict_str[ordered_terms[14][n][1]]\
         for n in range(len(ordered_terms[14]))])
        f15 = np.sum([ordered_terms[15][n][0]*init_dict_str[ordered_terms[15][n][1]]\
         for n in range(len(ordered_terms[15]))])
        f16 = np.sum([ordered_terms[16][n][0]*init_dict_str[ordered_terms[16][n][1]]\
         for n in range(len(ordered_terms[16]))])
        f17 = np.sum([ordered_terms[17][n][0]*init_dict_str[ordered_terms[17][n][1]]\
         for n in range(len(ordered_terms[17]))])
        f18 = np.sum([ordered_terms[18][n][0]*init_dict_str[ordered_terms[18][n][1]]\
         for n in range(len(ordered_terms[18]))])
        f19 = np.sum([ordered_terms[19][n][0]*init_dict_str[ordered_terms[19][n][1]]\
         for n in range(len(ordered_terms[19]))])
        f20 = np.sum([ordered_terms[20][n][0]*init_dict_str[ordered_terms[20][n][1]]\
         for n in range(len(ordered_terms[20]))])
        f21 = np.sum([ordered_terms[21][n][0]*init_dict_str[ordered_terms[21][n][1]]\
         for n in range(len(ordered_terms[21]))])
        f22 = np.sum([ordered_terms[22][n][0]*init_dict_str[ordered_terms[22][n][1]]\
         for n in range(len(ordered_terms[22]))])
        f23 = np.sum([ordered_terms[23][n][0]*init_dict_str[ordered_terms[23][n][1]]\
         for n in range(len(ordered_terms[23]))])
        return [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,\
        f18,f19,f20,f21,f22,f23]

def initial_chain(init, max_time = 100, sample_rate = 100):
    '''
    solves by solving each equation in the decay chain and for each time step
    and odein will give #of atoms of each species in list and repeate, which is
    reason for using splicing type
    
    --Returns--
    solns: list of amount of each isotope for each splice of time
    spec_names: list of names of corresponding species for solns
    '''
    time = np.linspace(0, max_time, max_time * sample_rate + 1)
    soln  = odeint(decaychain, init, time)
    num_soln = len(soln[0])
    solns = [soln[:,n] for n in range(num_soln)]
    spec_names = [init_strings[n][:-1] for n in range(num_soln)]
    return solns, spec_names

### Analyze solutions ###

def find_total_atoms(solns):
    '''
    Adds the atoms of all solutions in set of isotopes to check if the number of 
    total atoms is constant as expected
    '''
    total_atoms = []
    i = 0
    for i in range(len(solns[0])):
        local_total = 0
        for sol in solns:
            local_total += sol[i]
        total_atoms.append(local_total)
        i+=1
    return total_atoms

def find_branching_ratio_parent(init_strings, ordered_terms):
    '''
    Finds the branching ratio of parent -> daughter for each species in the chain
    '''
    parent_dict = find_parents(init_strings, ordered_terms)
    branching_ratio_dict = {}
    for spec in parent_dict:
        branching_ratios = []
        for parent in parent_dict[spec]:
            parent = make_string_iso_into_tuple(parent)
            daughters = find_daughters(parent+(1,))
            for daughter in daughters:
                if daughter != None and \
                tuple(list(daughter)[:-1]) == make_string_iso_into_tuple(spec):
                    br = daughter[-1]
                    branching_ratios.append(br)
        branching_ratio_dict.update({spec:branching_ratios})
    return branching_ratio_dict

def make_init_loc_dict(init_strings):
    '''
    Finds the location of each species in the init_strings
    '''
    n = 0
    init_loc_dict = {}
    for species in init_strings:
        init_loc_dict.update({species[:-1]:n})
        n+=1
    return init_loc_dict

def make_parent_loc_dict(init_strings, ordered_terms):
    '''  
    Combines parent dictionary and location dictionary to show the parent locations
    of each solution array
    
    --Returns--
    parent_loc_dict: dictionary of location of parent species and branching ratio in
    format {daughter: [ [parent1_loc, parent2_loc,...],
    [parent1->daugher branching ratio, parent2->daughter branching ratio,...]],...}
    '''
    parent_dict = find_parents(init_strings, ordered_terms)
    branching_ratio_dict = find_branching_ratio_parent(init_strings, ordered_terms)
    init_loc_dict = make_init_loc_dict(init_strings)
    parent_loc_dict = {}
    for species in parent_dict:
        species_loc = init_loc_dict[species]
        parent_locs = []
        if len(parent_dict[species]) > 0:
            for parent in parent_dict[species]:
                parent_locs.append(init_loc_dict[parent])
        parent_loc_dict.update({species_loc:[parent_locs,branching_ratio_dict[species]]})
    for species in parent_loc_dict: #if parent if upsteam. FIX THIS EVENTUALLY
        for parent_loc in parent_loc_dict[species][0]:
            if parent_loc > species:
                parent_loc_loc = parent_loc_dict[species][0].index(parent_loc)
                del parent_loc_dict[species][0][parent_loc_loc]
                del parent_loc_dict[species][1][parent_loc_loc]
    return parent_loc_dict
            
def find_decayed_atoms(solns, parent_loc_dict):
    '''
    Finds the number of decayed atoms from a list of list of solutions.
    Finds the difference in number of atoms first in initial species in the chain
    then uses that to calculate the number of decayed atome in later species. Does
    this by adding the number of the parent species to the daugherter with the 
    branching ratio factor then adds this to the difference in atoms of that daughter
    for each time point in the solns.
    
    --Inputs--
    solns: list of lists of solutions for each species in the chain at each time point
    parent_loc_dict: dictionary of the location of the parent of each species in the
    solns. Format is {daughter: [ [parent1_loc, parent2_loc,...],
    [parent1->daugher branching ratio, parent2->daughter branching ratio,...]],...}
    
    --Returns--
    diff_solns: list of lists of the number of decayed atoms for each species in same
    order as solns at each data point.
    '''
    diff_solns = []
    sol_len = len(solns[0])
    n = 1
    sol_diffs = [0]
    while n < sol_len: #Do first isotope in chain first
        current_diff = solns[0][n-1] - solns[0][n]
        sol_diffs.append(current_diff)
        n+=1
        
    diff_solns.append(sol_diffs)
    sol_num = 1 #track which isotope in chain
    for sol in solns[1:]:
        sol_diffs = [0] #always start with 0 because 0 diff at time 0
        n = 1 #start at second time
        while n < sol_len: #do full length of soln
            current_diff = 0
            
            longest_parent = 0
            for item in parent_loc_dict[sol_num]:
                if len(item) > longest_parent: longest_parent = len(item)
                
            for parent_num in range(longest_parent):
                parent_loc = parent_loc_dict[sol_num][0][parent_num]
                branching_ratio = parent_loc_dict[sol_num][1][parent_num]
                current_diff += branching_ratio * diff_solns[parent_loc][n]
            current_diff += sol[n-1] - sol[n]
            sol_diffs.append(current_diff)
            n+=1
        diff_solns.append(sol_diffs)
        sol_num += 1
    return diff_solns
       
def calc_total_decayed_atoms(diff_solns):
    '''
    Uses the solutions of decayed atoms to add up to the total number of decayed atoms
    for the entire time decay is examined for.
    
    --Returns--
    total_decay_solns: list of total number of decayed atoms in the order of decay
    ''' 
    total_decay_solns = []
    for decayed_atoms in diff_solns:
        total_decay = np.sum(decayed_atoms)
        total_decay_solns.append(total_decay)
    return total_decay_solns

def calc_radiation_energy_upper_limit(total_decay_solns, init_strings, joules = True):
    '''
    Calculates the upper limit on the exposure using the total number of decayed atoms
    and the amount of energy released by the decaying of the single atom. This is 
    calculated for each type of radiation.
    
    --Returns--
    rad_energy_dict: a dictionary of the type of radiation and the amount of energy
    associated with it. {beta-: xxxx, p: xxxx, ...}
    '''
    init_energy = 0
    rad_energy_dict = {'beta-':0, '2beta-':0, 'beta-n':0, 'beta-2n':0, 'beta-3n':0,\
    'beta-4n':0, 'beta-alpha':0, 'alpha':0, 'p':0, '2p':0, 'n':0, '2n':0,\
    'EC/beta+':0, 'EC/beta+p':0, 'EC/beta+2p':0, 'EC/beta+alpha':0}
    
    iso_order = [make_string_iso_into_tuple(iso_str[:-1]) for iso_str in init_strings]
    for total_decay, iso in zip(total_decay_solns, iso_order):
        z,a,isomer = iso
        iso_info = data[iso]
        if len(iso_info) > 3: #has decays
            number_of_decays = len(iso_info) - 3
            n = 3
            while n - 3 < number_of_decays:
                current_rad_energy = rad_energy_dict[iso_info[n][0]]
                decay_type = iso_info[n][0]
                decay_energy = total_decay * iso_info[n][2] * iso_info[n][4]
                if decay_type == 'alpha':
                    print(decay_energy)
                    decay_energy = decay_energy * ( (a-4.)/a ) 
                    print(decay_energy)
                    #account for nucleus recoil
                if init_energy == 0:
                    new_energy = decay_energy
                    init_energy = 1
                else:
                    new_energy = decay_energy + current_rad_energy
                rad_energy_dict.update({iso_info[n][0]:new_energy})
                n += 1
    if joules:
        for radiation in rad_energy_dict:
            mev_energy = rad_energy_dict[radiation]
            j_energy = kev_to_j(mev_energy)
            rad_energy_dict.update({radiation:j_energy})
    return rad_energy_dict
    
if __name__ == '__main__':
    data = create_isotope_data() #compile all isotope data
    
    modify_t12, isotope, max_time, normalized_dates, initial_mass, multiple_dates,\
    masses, plot_type = data_input() #collect initial isotope
    z,a,isomer,prob = isotope
        
    master = []
    find_decay(isotope) #populates the master list
    all_species_in_order = create_decay_order(master) #decay chain
    new_master_list = add_timecons(master,app_t12 = False,  modify_t12 = modify_t12)
    
    try:
        sample_rate = raw_input('Sample Frequency (samples/time scale): ')
        sample_rate = float(sample_rate)
    except:
        raise Exception('{} is not a valid sample_rate'.format(sample_rate))
    
    equations = equation_builder(new_master_list)
    master_terms = {}
    for item in sorted(equations):
        new_eq = {item: process_equation(equations[item])}
        master_terms.update(new_eq) #finishes building equations
        
    init_dict = create_init_dict((z,a,isomer), initial_mass, master_terms)
    init_strings = ['({},{},{})i'.format(elm[0],elm[1],elm[2]) for \
    elm in all_species_in_order] #value for each isotope in decay chain
    
    init = [init_dict[elm] for elm in init_strings]
    ordered_terms = create_ordered_terms(init_strings, master_terms)
    parent_loc_dict = make_parent_loc_dict(init_strings, ordered_terms)
    time = np.linspace(0, max_time, max_time * sample_rate + 1)
    solns, spec_names = initial_chain(init, max_time=max_time, sample_rate=sample_rate)
    
    if multiple_dates:
        for date, add_mass in zip(normalized_dates[1:],masses[1:]):
            start_loc = int(np.ceil(sample_rate * date))
            init = [sol[start_loc] for sol in solns]
            init[0] += add_mass
            new_max_time = max_time - date
            new_solns, spec_names = initial_chain(init, max_time=new_max_time,\
             sample_rate=sample_rate)
            solns = [np.append(sol[:start_loc], new_sol) for sol,new_sol in\
             zip(solns, new_solns)]
    
    diff_solns = find_decayed_atoms(solns, parent_loc_dict)
    branching_ratio_dict = find_branching_ratio_parent(init_strings, ordered_terms)
    activity_solns = convert_all_list_to_activity(solns, [eval(i) for i in spec_names])
    
    total_decay_solns = calc_total_decayed_atoms(diff_solns)
    rad_energy_dict = calc_radiation_energy_upper_limit(total_decay_solns, init_strings)
    
    print_exp_dict(rad_energy_dict) #prints radiation energies
    save_solns(z,a,isomer, modify_t12, spec_names, solns, time) #saves solutions to text
    create_plots(plot_type, solns, spec_names) #creates plots of solutions