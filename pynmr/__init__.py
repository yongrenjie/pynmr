import numpy as np
from pathlib import Path

from . import pulse

def readfid(path):
    path = str(path) # np.fromfile doesn't take Path objects
    fid = np.fromfile(path, dtype=np.int32)
    td = fid.size
    fid = fid.reshape(int(td/2), 2)
    fid = np.transpose(fid) # so now fid[0] is the real part, fid[1] the imaginary
    # TODO: Understand Bruker group delay
    return fid[0] + (1j * fid[1])
    

def read1d(path):
    path = str(path)
    return np.fromfile(path, dtype=np.int32)


def scale1d(spec, procs):
    ncproc = int(getpar(procs, "NC_proc"))
    return spec * (2 ** ncproc)


def readscale1d(path):
    spec = Path(path) / "1r"
    spec = str(spec)
    x = np.fromfile(spec, dtype=np.int32)
    procs = Path(path) / "procs"
    return scale1d(x, procs)


def getpar(path, par):
    # Remove any spaces from par
    if len(par.split()) > 1:
        par = "".join(par.split())

    parshort = par.rstrip("1234567890")
    longstr = "##$" + par.lower() + "="
    ans = ""
    
    # If there are no numbers in par
    if parshort == par:
        with open(path, 'r') as parfile:
            for line in parfile:
                if line.lower().startswith(longstr):
                    # usual case, like TD or SI
                    ans = line.split(' ', 2)[1].rstrip()
                    # if user wants the entire list, for whatever reason
                    if ans[0] == "(" and ans[-1] == ")":
                        l = parfile.readline()
                        ans = ""
                        while not l.startswith("##"):
                            ans = ans + l.rstrip() + " "
                            l = parfile.readline()
                        ans = ans.rstrip()
                    break
    # If there are numbers in par
    else:
        shortstr = "##$" + parshort.lower() + "="
        with open(path, 'r') as parfile:
            for line in parfile:
                # things like SFO1
                if line.lower().startswith(longstr):
                    ans = line.split(' ', 2)[1].rstrip()
                    break
                # things like CNST20
                elif line.lower().startswith(shortstr):
                    l = parfile.readline()
                    ans = ""
                    while not l.startswith("##"):
                        ans = ans + l.rstrip() + " "
                        l = parfile.readline()
                    ans = ans.rstrip().split()
                    num = int(par[len(parshort):]) # CNST20 -> 20
                    ans = ans[num] # TS lists start from 0.
                    break
    if ans == "":
        print("Parameter {} not found.".format(par))
    return ans.strip("<>")


def wparfile(dir, expno, par, val):
    p = Path(dir) / str(expno)
    # convert non-lists to lists
    if not isinstance(par, (list, np.ndarray)):
        par = [par]
    if not isinstance(val, (list, np.ndarray)):
        val = [val]
    if len(par) != len(val):
        raise IndexError("Parameter and value arrays are not the same length.")

    if not p.parent.is_dir():
        raise OSError("The specified folder {} does not exist.".format(dir))
    else:
        text = "".join(["{} : {}\n".format(par[i], val[i]) for i in range(len(par))])
        p.write_text(text)
        return 1

